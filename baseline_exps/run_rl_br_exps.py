"""Script for running BR-PPO evals against agent populations."""
import argparse
import csv
import math
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from typing import Callable, Dict, Optional, Tuple

import exp_utils
import numpy as np
import posggym
import torch
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.wrappers import FlattenObservations

from posggym_baselines.ppo.br_ppo import BRPPOConfig
from posggym_baselines.ppo.network import PPOModel
from posggym_baselines.utils import NoOverwriteRecordVideo, strtobool


def get_env_creator_fn(
    config: BRPPOConfig, env_idx: int, worker_idx: Optional[int] = None
) -> Callable:
    """Get function for creating the environment."""

    def thunk():
        capture_video = (
            config.capture_video
            and env_idx == 0
            and (worker_idx is None or worker_idx == 0)
        )
        render_mode = "rgb_array" if capture_video else None
        env = posggym.make(config.env_id, render_mode=render_mode, **config.env_kwargs)
        if capture_video:
            eval_episodes_per_env = math.ceil(
                config.eval_episodes / config.num_eval_envs
            )
            env = NoOverwriteRecordVideo(
                env,
                config.video_dir,
                episode_trigger=(
                    lambda ep: ep % max(1, eval_episodes_per_env // 10) == 0
                ),
            )
        env = AgentEnvWrapper(env, config.get_other_agent_fn())
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def run_evaluation_episodes(
    br_policy: Tuple[str, PPOModel],
    config: BRPPOConfig,
) -> Dict[str, float]:
    """Run evaluation episodes for a single BR-PPO policy."""
    assert config.num_agents == 1
    num_envs, num_agents = config.num_eval_envs, config.num_agents
    device = config.eval_device
    eval_episodes_per_env = math.ceil(config.eval_episodes / num_envs)

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

    num_dones = np.zeros((num_envs, 1))
    per_env_return = np.zeros((num_envs, num_agents))
    per_env_disc_return = np.zeros((num_envs, num_agents))

    timesteps = np.zeros((num_envs, 1))
    ep_returns = []
    ep_disc_returns = []

    while any(k < eval_episodes_per_env for k in num_dones):
        with torch.no_grad():
            for i, policy in enumerate([br_policy]):
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
        next_done = torch.Tensor(agent_dones).reshape((num_envs, num_agents)).to(device)

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

    env.close()
    mean_ep_returns = np.mean(ep_returns, axis=0)
    std_ep_returns = np.std(ep_returns, axis=0)
    mean_ep_disc_returns = np.mean(ep_disc_returns, axis=0)
    std_ep_disc_returns = np.std(ep_disc_returns, axis=0)

    return {
        "num_episodes": len(ep_returns),
        "mean_returns": mean_ep_returns[0],
        "std_returns": std_ep_returns[0],
        "mean_discounted_returns": mean_ep_disc_returns[0],
        "std_discounted_returns": std_ep_disc_returns[0],
    }


def main(args):
    torch.set_num_threads(1)
    env_data = exp_utils.get_env_data(None, None, full_env_id=args.full_env_id)
    env_data.pprint()
    print()

    config_kwargs = deepcopy(exp_utils.DEFAULT_PPO_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": get_env_creator_fn,
            "env_id": env_data.env_kwargs["env_id"],
            "env_kwargs": env_data.env_kwargs["env_kwargs"],
            "eval_device": "cuda" if args.cuda else "cpu",
            "track_wandb": False,
            "disable_logging": True,
            "eval_episodes": args.num_episodes,
        }
    )
    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    configs = {
        "P0": BRPPOConfig(
            exp_name=f"BR-PPO_{args.full_env_id}_P0_eval",
            other_agent_ids=env_data.agents_P0,
            **config_kwargs,
        ),
        "P1": BRPPOConfig(
            exp_name=f"BR-PPO_{args.full_env_id}_P1_eval",
            other_agent_ids=env_data.agents_P1,
            **config_kwargs,
        ),
    }

    results_file = exp_utils.RESULTS_DIR / (
        f"BR-PPO_{args.full_env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    result_headers = [
        "env_id",
        "train_seed",
        "train_pop",
        "eval_pop",
        "num_episodes",
        "mean_returns",
        "std_returns",
        "mean_discounted_returns",
        "std_discounted_returns",
    ]
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_headers)
        writer.writeheader()

    for train_pop, test_pop in product(["P0", "P1"], ["P0", "P1"]):
        train_config = configs[train_pop]
        test_config = configs[test_pop]
        br_policy = train_config.load_policies(device=test_config.eval_device)["BR"]
        br_policy.eval()
        for seed, br_model_file in env_data.br_model_files[train_pop].items():
            print(f"Running BR-PPO eval for {train_pop} -> {test_pop} (seed {seed})")
            checkpoint = torch.load(br_model_file, map_location=test_config.eval_device)
            br_policy.load_state_dict(checkpoint["model"])
            eval_start_time = time.time()
            results = run_evaluation_episodes(br_policy, test_config)
            eval_time = time.time() - eval_start_time
            print(f"Eval time: {eval_time:.2f}s")

            with open(results_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result_headers)
                writer.writerow(
                    {
                        "env_id": env_data.env_id,
                        "train_seed": seed,
                        "train_pop": train_pop,
                        "eval_pop": test_pop,
                        **results,
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--full_env_id",
        type=str,
        required=True,
        choices=[
            "CooperativeReaching-v0",
            "Driving-v1",
            "LevelBasedForaging-v3",
            "PredatorPrey-v0",
            "PursuitEvasion-v1_i0",
            "PursuitEvasion-v1_i1",
            "all",
        ],
        help="Name of environment to train on.",
    )
    parser.add_argument(
        "--capture_video",
        type=strtobool,
        default=False,
        help="Whether to capture videos of the environment during training.",
    )
    parser.add_argument(
        "--cuda",
        type=strtobool,
        default=True,
        help="Whether to use CUDA for learner if available.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate.",
    )
    args = parser.parse_args()

    if args.full_env_id == "all":
        print("Running all envs")
        for full_env_id in exp_utils.load_all_env_data():
            args.full_env_id = full_env_id
            print()
            main(args)
    else:
        main(args)
