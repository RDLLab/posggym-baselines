"""Functions for PPO population evaluation."""
import math
import time
from itertools import product
from multiprocessing.queues import Empty
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from posggym_baselines.ppo.network import PPOModel


if TYPE_CHECKING:
    from posggym_baselines.ppo.config import PPOConfig


# Define a type for the evaluation function.
# Arguments:
# - list of policy dictionaries (one list per agent in the environment)
# - config and returns a
# Returns:
# - dictionary mapping from eval metric key to the value. The value can be a NxM
#     matrix, a scalar, or matplotlib Figure
EvalFn = Callable[
    [Dict[str, PPOModel], "PPOConfig"],
    Dict[str, Union[np.ndarray, float, plt.Figure]],
]

# actions, rewards, done, obs
Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_eval_worker(
    config: "PPOConfig",
    recv_queue: mp.Queue,
    send_queue: mp.Queue,
    terminate_event: mp.Event,
):
    """Worker function for running evaluations.

    Arguments
    ---------
    config: BaseConfig
        The configuration for this rollout worker.
    recv_queue: mp.Queue
        The queue from which to receive the policy weights, etc from the learner.
    send_queue: mp.Queue
        The queue to which to send the collected trajectories to the learner.
    terminate_event: mp.Event
        The event which signals the rollout worker to terminate.

    """
    print(f"eval: Started, Device={config.eval_device}")

    # Limit worker to using a single CPU thread.
    # This prevents each worker from using all available cores, which can
    # lead to each worker being slower due to contention with rollout workers
    torch.set_num_threads(1)

    # policy setup
    policies = config.load_policies(device=config.eval_device)

    print("eval: Starting evaluation loop.")
    while not terminate_event.is_set():
        # wait for latest weights from learner
        policy_weights = {}
        global_step = None
        while not terminate_event.is_set():
            try:
                (policy_weights, global_step) = recv_queue.get(timeout=1)
                break
            except Empty:
                pass

        if terminate_event.is_set():
            break

        eval_start_time = time.time()
        assert len(policy_weights) == len(policies)
        for policy_id, policy in policies.items():
            policy.load_state_dict(policy_weights[policy_id])

        # delete reference to learner weights to free any shared CUDA memory reference
        del policy_weights

        # run evaluation
        print(f"eval: running evaluation for {global_step=}.")
        all_metrics = {}
        for eval_fn in config.eval_fns:
            metrics = eval_fn(policies, config)
            all_metrics.update(metrics)

        eval_time = time.time() - eval_start_time
        print(f"eval: evaluation for {global_step=} finished, {eval_time:.2f} s.")
        # send results back to learner
        send_queue.put(
            {"metrics": all_metrics, "global_step": global_step, "eval_time": eval_time}
        )

    print("eval: Terminated.")


def run_pairwise_evaluation(
    policies: List[Dict[str, PPOModel]], config: "PPOConfig"
) -> Dict[str, np.ndarray]:
    """Run pairwise evaluation of policy population.

    Note, this function is only defined for environments with two agents.

    pop_size1 = len(policies[0])
    pop_size2 = len(policies[1])

    Arguments
    ---------
    policies:
        A list of dicts containing the policies to evaluate. List length must equal the
        number of agents in the environment. Each dict should map from "policy_id" to
        a PPOModel.
    config:
        The PPO configuration.

    Returns
    -------
    metrics:
        A dictionary mapping from pairwise return type name to the pairwise return
        matrix. Contains entries for the average returns, average discounted returns,
        as well as their standard deviations.
    """
    assert len(policies) == 2
    assert config.num_agents == 2
    num_envs, num_agents = config.num_eval_envs, config.num_agents
    device = config.eval_device

    eval_episodes_per_env = math.ceil(config.eval_episodes / num_envs)
    print(f"eval: Running pairwise evaluation {num_envs=} {eval_episodes_per_env=}")

    policy_ids1 = list(policies[0])
    policy_ids2 = list(policies[1])

    pairwise_returns1 = np.zeros((2, len(policy_ids1), len(policy_ids2)))
    pairwise_disc_returns1 = np.zeros((2, len(policy_ids1), len(policy_ids2)))
    pairwise_returns2 = np.zeros((2, len(policy_ids2), len(policy_ids1)))
    pairwise_disc_returns2 = np.zeros((2, len(policy_ids2), len(policy_ids1)))

    for (policy_idx1, policy_id1), (policy_idx2, policy_id2) in product(
        enumerate(policy_ids1), enumerate(policy_ids2)
    ):
        print(f"- Running evaluation for {policy_id1} vs {policy_id2}")
        policy1 = policies[0][policy_id1]
        policy2 = policies[1][policy_id2]

        env = config.load_vec_env(num_envs=num_envs)
        next_obs = (
            torch.tensor(env.reset()[0])
            .float()
            .to(device)
            .reshape(num_envs, num_agents, -1)
        )
        next_action = torch.zeros((num_envs, num_agents)).long().to(device)
        next_done = torch.zeros((num_envs, num_agents)).to(device)
        next_lstm_state = (
            torch.zeros(
                (config.lstm_num_layers, num_envs, num_agents, config.lstm_size)
            ).to(device),
            torch.zeros(
                (config.lstm_num_layers, num_envs, num_agents, config.lstm_size)
            ).to(device),
        )

        # Initialize objects to track returns.
        num_dones = np.zeros((num_envs, 1))
        per_env_return = np.zeros((num_envs, num_agents))
        per_env_disc_return = np.zeros((num_envs, num_agents))

        timesteps = np.zeros((num_envs, 1))
        ep_returns = []
        ep_disc_returns = []

        while any(k < eval_episodes_per_env for k in num_dones):
            with torch.no_grad():
                for i, policy in enumerate([policy1, policy2]):
                    obs_i = next_obs[:, i, :]
                    done_i = next_done[:, i]
                    lstm_state_i = (
                        next_lstm_state[0][:, :, i, :],
                        next_lstm_state[1][:, :, i, :],
                    )
                    actions_i, _, _, lstm_state_i = policy.get_action(
                        obs_i, lstm_state_i, done_i
                    )

                    next_action[:, i] = actions_i
                    if lstm_state_i is not None:
                        next_lstm_state[0][:, :, i] = lstm_state_i[0]
                        next_lstm_state[1][:, :, i] = lstm_state_i[1]

            next_obs, rews, terms, truncs, dones, _ = env.step(
                next_action.reshape(-1).cpu().numpy()
            )
            agent_dones = terms | truncs

            next_obs = (
                torch.Tensor(next_obs)
                .reshape((num_envs, num_agents, -1))
                .float()
                .to(device)
            )
            next_done = (
                torch.Tensor(agent_dones).reshape((num_envs, num_agents)).to(device)
            )

            rews = rews.reshape((num_envs, num_agents))

            # Log per thread returns
            per_env_disc_return += config.gamma**timesteps * rews
            per_env_return += rews
            timesteps = timesteps + 1

            for env_idx, env_done in enumerate(dones):
                if env_done:
                    timesteps[env_idx] = 0
                    num_dones[env_idx] += 1
                    ep_returns.append(per_env_return[env_idx].copy())
                    ep_disc_returns.append(per_env_disc_return[env_idx].copy())
                    per_env_return[env_idx] = 0
                    per_env_disc_return[env_idx] = 0

        mean_ep_returns = np.mean(ep_returns, axis=0)
        std_ep_returns = np.std(ep_returns, axis=0)
        pairwise_returns1[0, policy_idx1, policy_idx2] = mean_ep_returns[0]
        pairwise_returns2[0, policy_idx2, policy_idx1] = mean_ep_returns[1]
        pairwise_returns1[1, policy_idx1, policy_idx2] = std_ep_returns[0]
        pairwise_returns2[1, policy_idx2, policy_idx1] = std_ep_returns[1]

        mean_ep_disc_returns = np.mean(ep_disc_returns, axis=0)
        std_ep_disc_returns = np.std(ep_disc_returns, axis=0)
        pairwise_disc_returns1[0, policy_idx1, policy_idx2] = mean_ep_disc_returns[0]
        pairwise_disc_returns2[0, policy_idx2, policy_idx1] = mean_ep_disc_returns[1]
        pairwise_disc_returns1[1, policy_idx1, policy_idx2] = std_ep_disc_returns[0]
        pairwise_disc_returns2[1, policy_idx2, policy_idx1] = std_ep_disc_returns[1]
        env.close()

    print("\neval: pairwise evaluation finished.")
    print("policy_ids1", policy_ids1)
    print("policy_ids2", policy_ids2)
    print("pairwise_returns1", pairwise_returns1[0])

    return {
        "pairwise_returns1": pairwise_returns1[0],
        "pairwise_returns_std1": pairwise_returns1[1],
        "pairwise_disc_returns1": pairwise_disc_returns1[0],
        "pairwise_disc_returns_std1": pairwise_disc_returns1[1],
        "pairwise_returns2": pairwise_returns2[0],
        "pairwise_returns_std2": pairwise_returns2[1],
        "pairwise_disc_returns2": pairwise_disc_returns2[0],
        "pairwise_disc_returns_std2": pairwise_disc_returns2[1],
    }


def run_all_pairwise_evaluation(
    policies: Dict[str, PPOModel], config: "PPOConfig"
) -> Dict[str, np.ndarray]:
    """Run pairwise evaluation for all pairs in a policy population.

    Note, this is the same as run_pairwise_evaluation but follows EvalFn protocol.
    """
    return run_pairwise_evaluation([policies, policies], config)


def run_train_distribution_evaluation(
    policies: Dict[str, PPOModel], config: "PPOConfig"
) -> np.ndarray:
    """Run pairwise evaluation for training distribution pairs in policy population.

    Assumes env is symmetric.
    """
    policy_ids = list(policies)
    pw_returns = np.zeros((4, len(policies), len(policies)))
    for idx_i, policy_id in enumerate(policies):
        partner_dist = config.get_policy_partner_distribution(policy_id)
        results = run_pairwise_evaluation(
            [
                {policy_id: policies[policy_id]},
                {pi_id: policies[pi_id] for pi_id in partner_dist},
            ],
            config,
        )
        for idx, partner_id in enumerate(partner_dist):
            idx_j = policy_ids.index(partner_id)
            pw_returns[0, idx_i, idx_j] = results["pairwise_returns1"][0, idx]
            pw_returns[1, idx_i, idx_j] = results["pairwise_returns_std1"][0, idx]
            pw_returns[2, idx_i, idx_j] = results["pairwise_disc_returns1"][0, idx]
            pw_returns[3, idx_i, idx_j] = results["pairwise_disc_returns_std1"][0, idx]

    return pw_returns


def render_policies(
    policies: List[Dict[str, PPOModel]], num_episodes: int, env, config: "PPOConfig"
) -> Dict[str, np.ndarray]:
    """Render pairwise episodes of policy population.

    Arguments
    ---------
    policies:
        A list of dicts containing the policies to evaluate. List length must equal the
        number of agents in the environment. Each dict should map from "policy_id" to
        a PPOModel.
    config:
        The PPO configuration.
    """
    assert len(policies) == config.num_agents
    num_envs, num_agents = 1, config.num_agents
    device = config.eval_device

    for policy_ids in product(*policies):
        print(f"\nRendering policies: {policy_ids}")

        next_obs = (
            torch.tensor(env.reset()[0])
            .float()
            .to(device)
            .reshape(num_envs, num_agents, -1)
        )
        next_action = torch.zeros((num_envs, num_agents)).long().to(device)
        next_done = torch.zeros((num_envs, num_agents)).to(device)
        next_lstm_state = (
            torch.zeros(
                (config.lstm_num_layers, num_envs, num_agents, config.lstm_size)
            ).to(device),
            torch.zeros(
                (config.lstm_num_layers, num_envs, num_agents, config.lstm_size)
            ).to(device),
        )

        num_dones = np.zeros((num_envs, 1))
        per_env_return = np.zeros((num_envs, num_agents))
        per_env_disc_return = np.zeros((num_envs, num_agents))

        timesteps = np.zeros((num_envs, 1))
        ep_returns = []
        ep_disc_returns = []

        while num_dones.sum() < num_episodes:
            env.render()
            with torch.no_grad():
                for i, policy_id in enumerate(policy_ids):
                    policy = policies[i][policy_id]
                    obs_i = next_obs[:, i, :]
                    done_i = next_done[:, i]
                    lstm_state_i = (
                        next_lstm_state[0][:, :, i, :],
                        next_lstm_state[1][:, :, i, :],
                    )
                    actions_i, _, _, lstm_state_i = policy.get_action(
                        obs_i, lstm_state_i, done_i
                    )

                    next_action[:, i] = actions_i
                    if lstm_state_i is not None:
                        next_lstm_state[0][:, :, i] = lstm_state_i[0]
                        next_lstm_state[1][:, :, i] = lstm_state_i[1]

            next_obs, rews, terms, truncs, dones, _ = env.step(
                next_action.reshape(-1).cpu().numpy()
            )
            agents_done = terms | truncs
            next_obs = (
                torch.Tensor(next_obs)
                .reshape((num_envs, num_agents, -1))
                .float()
                .to(device)
            )
            next_done = (
                torch.Tensor(agents_done).reshape((num_envs, num_agents)).to(device)
            )

            rews = rews.reshape((num_envs, num_agents))

            per_env_disc_return += config.gamma**timesteps * rews
            per_env_return += rews
            timesteps = timesteps + 1

            for env_idx, flag in enumerate(dones):
                # If an episode in one of the environments ends
                if flag:
                    timesteps[env_idx] = 0
                    num_dones[env_idx] += 1
                    ep_returns.append(per_env_return[env_idx].copy())
                    ep_disc_returns.append(per_env_disc_return[env_idx].copy())
                    per_env_return[env_idx] = 0
                    per_env_disc_return[env_idx] = 0

        mean_ep_returns = np.mean(ep_returns, axis=0)
        std_ep_returns = np.std(ep_returns, axis=0)

        mean_ep_disc_returns = np.mean(ep_disc_returns, axis=0)
        std_ep_disc_returns = np.std(ep_disc_returns, axis=0)
        env.close()

        print(f"policy_ids={policy_ids}")
        for i, policy_id in enumerate(policy_ids):
            print(
                f"  {policy_id} "
                f"- return = {mean_ep_returns[i]:.2f} +/-  {std_ep_returns[i]:.2f} "
                f"- disc. return = {mean_ep_disc_returns[i]:.2f} "
                f"+/- {std_ep_disc_returns[i]:.2f}"
            )
