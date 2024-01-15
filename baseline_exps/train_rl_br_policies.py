"""Script for training BR policy against a population of policies.

The BR policy is trained using PPO, with the policies for the other agents in the
environment sampled from the population of policies at the start of each training
episode.
"""
import argparse
from copy import deepcopy
from pprint import pprint
from typing import Callable, Optional

import exp_utils
import posggym
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.wrappers import FlattenObservations, RecordVideo

from posggym_baselines.ppo.br_ppo import BRPPOConfig
from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.core import run_ppo
from posggym_baselines.utils import strtobool


def record_episode_trigger(ep):
    return ep % 200 == 0


def get_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: Optional[int] = None
) -> Callable:
    """Get function for creating the environment."""

    def thunk():
        capture_video = config.capture_video and env_idx == 0 and worker_idx == 0
        render_mode = "rgb_array" if capture_video else None
        env = posggym.make(config.env_id, render_mode=render_mode, **config.env_kwargs)
        if capture_video:
            env = RecordVideo(
                env, config.video_dir, episode_trigger=record_episode_trigger
            )

        env = AgentEnvWrapper(env, config.get_other_agent_fn())
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def train(args):
    env_data = exp_utils.get_env_data(None, None, full_env_id=args.full_env_id)
    env_kwargs = env_data.env_kwargs

    if args.other_agent_population == "P0":
        other_agent_policy_ids = env_data.agents_P0
    else:
        other_agent_policy_ids = env_data.agents_P1

    config_kwargs = deepcopy(exp_utils.DEFAULT_PPO_CONFIG)
    config_kwargs.update(
        {
            "exp_name": f"BR-PPO_{args.full_env_id}_{args.other_agent_population}",
            "env_creator_fn": get_env_creator_fn,
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = BRPPOConfig(
        other_agent_ids=other_agent_policy_ids,
        **config_kwargs,
    )
    pprint(config)
    run_ppo(config)


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
        ],
        help="Name of environment to train on.",
    )
    parser.add_argument(
        "--other_agent_population",
        type=str,
        required=True,
        choices=["P0", "P1"],
        help="Other agent population.",
    )
    parser.add_argument(
        "--track_wandb",
        type=strtobool,
        default=True,
        help="Whether to track the experiment with wandb.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional wandb group name.",
    )
    parser.add_argument(
        "--disable_logging",
        type=strtobool,
        default=False,
        help="Whether to disable all logging (for debugging).",
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
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=int(1e8),
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--use_lstm",
        type=strtobool,
        default=True,
        help="Whether to use LSTM based policy network for learner.",
    )
    train(parser.parse_args())
