"""Script for evaluating trained BR-PPO policies against a population of agents."""
import argparse
import math
import yaml
from pprint import pprint
from typing import Callable, Optional, Tuple, Dict
from copy import deepcopy

import numpy as np
import torch
import posggym
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.wrappers import FlattenObservations

from posggym_baselines.utils import strtobool, NoOverwriteRecordVideo
from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.br_ppo import BRPPOConfig
from posggym_baselines.ppo.network import PPOModel
from posggym_baselines.ppo.core import load_policies

from posggym_baselines.exps.grid_world.train_br_ppo import DEFAULT_CONFIG


def get_eval_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: Optional[int] = None
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


def load_config(args, train=True):
    with open(args.env_kwargs_file, "r") as f:
        env_kwargs = yaml.safe_load(f)

    if train:
        with open(args.train_agents_file, "r") as f:
            other_agent_policy_ids = yaml.safe_load(f)
    else:
        with open(args.test_agents_file, "r") as f:
            other_agent_policy_ids = yaml.safe_load(f)

    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "exp_name": args.exp_name + "_" + "train" if train else "test",
            "env_creator_fn": get_eval_env_creator_fn,
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
            "eval_device": "cuda" if args.cuda else "cpu",
            "track_wandb": False,
            "disable_logging": True,
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = BRPPOConfig(
        # BR-PPO specific config
        other_agent_ids=other_agent_policy_ids,
        **config_kwargs,
    )
    # pprint(config)
    return config


def run_evaluation_episodes(
    br_policy: Tuple[str, PPOModel],
    config: PPOConfig,
) -> Dict[str, float]:
    """Run evaluation episodes for a single BR-PPO policy."""
    assert config.num_agents == 1
    num_envs, num_agents = config.num_eval_envs, config.num_agents
    device = config.eval_device
    eval_episodes_per_env = math.ceil(config.eval_episodes / num_envs)
    print(f"eval: Running evaluation episodes {num_envs=} {eval_episodes_per_env=}")

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

    # compute 95% confidence intervals
    ci_ep_returns = 1.96 * std_ep_returns / np.sqrt(len(ep_returns))
    ci_ep_disc_returns = 1.96 * std_ep_disc_returns / np.sqrt(len(ep_disc_returns))

    return {
        "mean_ep_returns": mean_ep_returns[0],
        "std_ep_returns": std_ep_returns[0],
        "ci_ep_returns": ci_ep_returns[0],
        "mean_ep_disc_returns": mean_ep_disc_returns[0],
        "std_ep_disc_returns": std_ep_disc_returns[0],
        "ci_ep_disc_returns": ci_ep_disc_returns[0],
    }


def run_evaluation(args):
    assert args.train_agents_file is not None or args.test_agents_file is not None

    if args.train_agents_file is not None:
        config = load_config(args, train=True)
        policies = load_policies(
            config, args.saved_model_dir, device=config.eval_device
        )
        assert len(policies) == 1 and "BR" in policies
        eval_results = run_evaluation_episodes(policies["BR"], config)

        print("Training results")
        pprint(eval_results)

    if args.test_agents_file is not None:
        config = load_config(args, train=False)
        policies = load_policies(
            config, args.saved_model_dir, device=config.eval_device
        )
        assert len(policies) == 1 and "BR" in policies
        eval_results = run_evaluation_episodes(policies["BR"], config)

        print("Test results")
        pprint(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Name for experiment.",
    )

    parser.add_argument(
        "--saved_model_dir",
        type=str,
        required=True,
        help="Directory containing saved model to evaluate.",
    )
    parser.add_argument(
        "--env_kwargs_file",
        type=str,
        help="Path to YAML file containing env kwargs.",
    )
    parser.add_argument(
        "--train_agents_file",
        type=str,
        default=None,
        help="Path to YAML file containing training population of other agents.",
    )
    parser.add_argument(
        "--test_agents_file",
        type=str,
        default=None,
        help="Path to YAML file containing training population of other agents.",
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
    run_evaluation(args)
