"""Script for running BR-PPO evals against agent populations."""
import argparse
import csv
import os
import math
import yaml
from pprint import pprint
from itertools import product
from typing import Callable, Optional, Tuple, Dict
from copy import deepcopy

import numpy as np
import torch
import posggym
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.wrappers import FlattenObservations

from posggym_baselines.utils import strtobool, NoOverwriteRecordVideo
from posggym_baselines.ppo.br_ppo import BRPPOConfig
from posggym_baselines.ppo.network import PPOModel
from posggym_baselines.exps.grid_world.train_br_ppo import DEFAULT_CONFIG


ENV_DATA_DIR = os.path.join(os.path.dirname(__file__), "env_data")


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


def get_env_data(env_id):
    """Get the env data for the given env name."""
    env_data_path = os.path.join(ENV_DATA_DIR, env_id)
    env_kwargs_file = os.path.join(env_data_path, "env_kwargs.yaml")
    with open(env_kwargs_file, "r") as f:
        env_kwargs = yaml.safe_load(f)

    agents_P0_file = os.path.join(env_data_path, "agents_P0.yaml")
    with open(agents_P0_file, "r") as f:
        agents_P0 = yaml.safe_load(f)

    agents_P1_file = os.path.join(env_data_path, "agents_P1.yaml")
    with open(agents_P1_file, "r") as f:
        agents_P1 = yaml.safe_load(f)

    br_models_dir = os.path.join(env_data_path, "br_models")

    br_model_files = {"P0": {}, "P1": {}}
    for model_file_name in os.listdir(br_models_dir):
        model_name = model_file_name.replace(".pt", "")
        tokens = model_name.split("_")
        train_pop = tokens[0]
        seed = int(tokens[1].replace("seed", ""))
        br_model_files[train_pop][seed] = os.path.join(br_models_dir, model_file_name)

    results_file = os.path.join(env_data_path, "br_results.csv")

    return env_kwargs, agents_P0, agents_P1, br_model_files, results_file


def load_config(
    args,
    env_kwargs,
    other_agents,
):
    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": get_env_creator_fn,
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
            "eval_device": "cuda" if args.cuda else "cpu",
            "track_wandb": False,
            "disable_logging": True,
            "eval_episodes": args.num_episodes,
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = BRPPOConfig(
        # BR-PPO specific config
        other_agent_ids=other_agents,
        **config_kwargs,
    )
    return config


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
    env_kwargs, agents_P0, agents_P1, br_model_files, results_file = get_env_data(
        args.env_id
    )
    print("env_kwargs:")
    pprint(env_kwargs)
    print("agents_P0:")
    pprint(agents_P0)
    print("agents_P1:")
    pprint(agents_P1)
    pprint(br_model_files)

    configs = {
        "P0": load_config(args, env_kwargs, agents_P0),
        "P1": load_config(args, env_kwargs, agents_P1),
    }
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
        config = configs[train_pop]
        for seed, br_model_file in br_model_files[train_pop].items():
            print(f"Running BR-PPO eval for {train_pop} -> {test_pop} (seed {seed})")
            br_policy = config.load_policies(device=config.eval_device)["BR"]
            checkpoint = torch.load(br_model_file, map_location=config.eval_device)
            br_policy.load_state_dict(checkpoint["model"])
            br_policy.eval()
            results = run_evaluation_episodes(br_policy, config)

            with open(results_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result_headers)
                writer.writerow(
                    {
                        "env_id": args.env_id,
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
        "--env_id",
        type=str,
        required=True,
        help="Name of environment.",
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
    main(parser.parse_args())
