"""Implementation of rollout workers which collect trajectories for PPO training."""
from multiprocessing.queues import Empty

import torch
import torch.multiprocessing as mp

import posggym_baselines.ppo.utils as ppo_utils
from posggym_baselines.ppo.config import PPOConfig
import contextlib


def run_rollout_worker(
    worker_id: int,
    config: PPOConfig,
    recv_queue: mp.JoinableQueue,
    send_queue: mp.JoinableQueue,
    terminate_event: mp.Event,
):
    """Rollout worker function for collecting trajectories.

    The rollout worker collect batches of trajectories and sends them to the learner
    for training.

    At the start of each batch, the rollout worker receives which policies to use along
    with their weights from the learner.

    Arguments
    ---------
    worker_id
        The id of this rollout worker.
    config
        The configuration for this rollout worker.
    recv_queue
        The queue from which to receive the policy weights, etc from the learner.
    send_queue
        The queue to which to send the collected trajectories to the learner.
    terminate_event
        The event which signals the rollout worker to terminate.

    """
    print(f"Worker {worker_id} - Started, Device={config.worker_device}")
    device = config.worker_device

    # Limit each rollout worker to using a single CPU thread.
    # This prevents each rollout worker from using all available cores, which can
    # lead to each rollout worker being slower due to contention.
    torch.set_num_threads(1)

    # env setup
    # Note: SyncVectorEnv runs multiple-env instances serially.
    envs = config.load_vec_env(num_envs=config.num_envs, worker_idx=worker_id)

    # actor and critic setup
    policies = config.load_policies(device=device)

    # worker buffers
    buf_shape = (config.num_rollout_steps, config.num_envs, config.num_agents)
    policy_idx_buf = torch.zeros(buf_shape).long().to(device)
    obs_buf = torch.zeros(buf_shape + config.obs_space.shape).to(device)
    actions_buf = torch.zeros(buf_shape + config.act_space.shape).to(device)
    logprobs_buf = torch.zeros(buf_shape).to(device)
    rewards_buf = torch.zeros(buf_shape).to(device)
    # +1 for bootstrapped value
    dones_buf = torch.zeros((buf_shape[0] + 1, *buf_shape[1:])).to(device)
    values_buf = torch.zeros((buf_shape[0] + 1, *buf_shape[1:])).to(device)
    lstm_state_shape = (
        config.lstm_num_layers,
        config.num_envs,
        config.num_agents,
        config.lstm_size,
    )
    lstm_states_buf = (
        torch.zeros((config.num_rollout_steps,) + lstm_state_shape).to(device),
        torch.zeros((config.num_rollout_steps,) + lstm_state_shape).to(device),
    )

    # setup variables for tracking current step outputs
    next_vars_shape = (config.num_envs, config.num_agents)
    next_obs = (
        torch.Tensor(envs.reset()[0])
        .to(config.worker_device)
        .reshape(config.num_envs, config.num_agents, -1)
    )
    next_done = torch.zeros(next_vars_shape).to(config.worker_device)
    next_lstm_state = (
        torch.zeros(lstm_state_shape).to(config.worker_device),
        torch.zeros(lstm_state_shape).to(config.worker_device),
    )
    next_action = torch.zeros(next_vars_shape).long().to(config.worker_device)
    next_logprobs = torch.zeros(next_vars_shape).to(config.worker_device)
    next_values = torch.zeros(next_vars_shape).to(config.worker_device)

    # mapping from policy idx to policy id
    policy_idx_to_id = config.get_policy_idx_to_id_mapping()
    policy_id_to_idx = config.get_policy_id_to_idx_mapping()

    # sample policy ids for each env
    sampled_policies = [
        config.sample_episode_policies() for _ in range(config.num_envs)
    ]
    sampled_policy_idxs = torch.zeros(config.num_envs, config.num_agents).long()
    for env_idx, env_policies in enumerate(sampled_policies):
        for agent_idx, policy_id in enumerate(env_policies):
            sampled_policy_idxs[env_idx, agent_idx] = policy_id_to_idx[policy_id]

    # Get reference to learner weights
    policy_weights = {}
    while len(policy_weights) == 0 and not terminate_event.is_set():
        with contextlib.suppress(Empty):
            policy_weights = recv_queue.get(timeout=1)
    assert len(policy_weights) == len(config.train_policies)

    # start batch collection loop
    print(f"Worker {worker_id} - Starting batch collection loop.")
    while not terminate_event.is_set():
        # wait for signal from learner to start batch collection
        while not terminate_event.is_set():
            with contextlib.suppress(Empty):
                recv_queue.get(timeout=1)
                recv_queue.task_done()
                break
        if terminate_event.is_set():
            break

        # sync weights
        for policy_id, weights in policy_weights.items():
            policies[policy_id].load_state_dict(weights)

        # collect batch of experience
        policy_episode_stats = {pi_id: [] for pi_id in config.get_all_policy_ids()}
        num_episodes = 0
        for step in range(0, config.num_rollout_steps):
            obs_buf[step] = next_obs
            policy_idx_buf[step] = sampled_policy_idxs
            dones_buf[step] = next_done
            lstm_states_buf[0][step] = next_lstm_state[0]
            lstm_states_buf[1][step] = next_lstm_state[1]

            # sample next action
            for policy_idx, policy_id in policy_idx_to_id.items():
                idxs = sampled_policy_idxs == policy_idx
                if idxs.sum() == 0:
                    continue
                obs_i = next_obs[idxs]
                lstm_i = (
                    next_lstm_state[0][:, idxs, :],
                    next_lstm_state[1][:, idxs, :],
                )
                done_i = next_done[idxs]
                policy_i = policies[policy_id]
                with torch.no_grad():
                    (
                        actions_i,
                        logprobs_i,
                        _,
                        values_i,
                        lstm_i,
                    ) = policy_i.get_action_and_value(obs_i, lstm_i, done_i)

                next_logprobs[idxs] = logprobs_i
                next_values[idxs] = values_i.squeeze(1)
                if lstm_i is not None:
                    next_lstm_state[0][:, idxs, :] = lstm_i[0]
                    next_lstm_state[1][:, idxs, :] = lstm_i[1]

                next_action[idxs] = actions_i

            actions_buf[step] = next_action
            logprobs_buf[step] = next_logprobs
            values_buf[step] = next_values

            # execute step.
            next_obs, reward, terminated, truncated, dones, infos = envs.step(
                next_action.reshape(-1).cpu().numpy()
            )
            agent_dones = terminated | truncated

            rewards_buf[step] = (
                torch.tensor(reward)
                .reshape((config.num_envs, config.num_agents))
                .to(config.worker_device)
            )
            next_obs = (
                torch.Tensor(next_obs)
                .reshape((config.num_envs, config.num_agents, -1))
                .to(config.worker_device)
            )
            next_done = (
                torch.Tensor(agent_dones)
                .reshape((config.num_envs, config.num_agents))
                .to(config.worker_device)
            )

            for env_idx, env_done in enumerate(dones):
                if env_done:
                    num_episodes += 1
                    # reset lstm state when episode ends
                    next_lstm_state[0][:, env_idx, :, :] = 0
                    next_lstm_state[1][:, env_idx, :, :] = 0

                    # get episode stats
                    for agent_id, policy_id in zip(
                        envs.possible_agents, sampled_policies[env_idx]
                    ):
                        if "episode" not in infos[agent_id]:
                            continue
                        env_vec_idx = env_idx
                        policy_episode_stats[policy_id].append(
                            (
                                infos[agent_id]["episode"]["r"][env_vec_idx],
                                infos[agent_id]["episode"]["l"][env_vec_idx],
                                infos[agent_id]["episode"]["t"][env_vec_idx],
                            )
                        )

                    # reset policy id when episode ends
                    sampled_policies[env_idx] = config.sample_episode_policies()
                    for agent_idx, policy_id in enumerate(sampled_policies[env_idx]):
                        sampled_policy_idxs[env_idx, agent_idx] = policy_id_to_idx[
                            policy_id
                        ]

        # log episode stats
        policy_stats = {}
        for policy_id, stats in policy_episode_stats.items():
            if len(stats) == 0:
                continue
            stats = torch.tensor(stats, dtype=torch.float32)
            policy_stats[policy_id] = {
                "mean_episode_return": torch.mean(stats[:, 0]),
                "min_episode_return": torch.min(stats[:, 0]),
                "max_episode_return": torch.max(stats[:, 0]),
                "mean_episode_length": torch.mean(stats[:, 1]),
                "mean_episode_time": torch.mean(stats[:, 2]),
            }

        # bootstrap value for final entry of batch if not done
        for policy_idx, policy_id in policy_idx_to_id.items():
            idxs = sampled_policy_idxs == policy_idx
            if idxs.sum() == 0:
                continue
            obs_i = next_obs[idxs]
            lstm_i = (
                next_lstm_state[0][:, idxs, :],
                next_lstm_state[1][:, idxs, :],
            )
            done_i = next_done[idxs]
            policy_i = policies[policy_id]
            with torch.no_grad():
                values_i = policy_i.get_value(obs_i, lstm_i, done_i)

            next_values[idxs] = values_i.squeeze(1)
        values_buf[-1] = next_values
        dones_buf[-1] = next_done

        # Get sequence chunk indices for each batch dim
        shared_dones_buf = torch.any(dones_buf[:-1], dim=-1).long()
        seq_idxs, overlapping_seq_idxs = ppo_utils.get_seq_idxs(
            shared_dones_buf, config.seq_len
        )

        # split batch into sequences and pad to max_seq_len
        b_obs = ppo_utils.split_and_pad_batch(obs_buf, seq_idxs, config.seq_len)
        b_policy_idxs = ppo_utils.split_and_pad_batch(
            policy_idx_buf, seq_idxs, config.seq_len
        )
        b_actions = ppo_utils.split_and_pad_batch(actions_buf, seq_idxs, config.seq_len)
        b_logprobs = ppo_utils.split_and_pad_batch(
            logprobs_buf, seq_idxs, config.seq_len
        )
        b_rewards = ppo_utils.split_and_pad_batch(rewards_buf, seq_idxs, config.seq_len)
        # +1 to include additional step at end of sequence which is used for calculating
        # advantages and returns
        # Also pad dones with 1.0, since this has the effect of zeroing out the
        # further steps in the sequence when calculating the advantages and returns
        b_dones = ppo_utils.split_and_pad_batch(
            dones_buf, overlapping_seq_idxs, config.seq_len + 1, padding_value=1.0
        )
        b_values = ppo_utils.split_and_pad_batch(
            values_buf, overlapping_seq_idxs, config.seq_len + 1
        )

        # The LSTM state buffer stores the state for each transition in the batch,
        # but only the initial LSTM state for each sequence chunk is used by the learner
        # during update, so we need to split the LSTM state buffer by sequences
        # and keep only the initial LSTM state for each sequence chunk.
        b_initial_lstm_states = (
            ppo_utils.split_lstm_state_batch(lstm_states_buf[0], seq_idxs),
            ppo_utils.split_lstm_state_batch(lstm_states_buf[1], seq_idxs),
        )

        # send batch of data to learner
        send_queue.put(
            {
                "obs": b_obs,
                "policy_idxs": b_policy_idxs,
                "actions": b_actions,
                "logprobs": b_logprobs,
                "rewards": b_rewards,
                "dones": b_dones,
                "values": b_values,
                "initial_lstm_states": b_initial_lstm_states,
                "policy_stats": policy_stats,
            }
        )

    print(f"Worker {worker_id} - Termination signal received.")
    envs.close()

    print(f"Worker {worker_id} - Releasing shared resources.")
    del policy_weights
    recv_queue.task_done()

    print(f"Worker {worker_id} - Waiting for shared resources to be released.")
    send_queue.join()

    print(f"Worker {worker_id} - Terminated.")
