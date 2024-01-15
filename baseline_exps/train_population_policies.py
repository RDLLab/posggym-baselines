"""Script for training population RL policies for the experiment environments.

Each population policy is trained using PPO. Supports both SP and KLR training with
and without best-response (BR) policies.

The number of policies in the population is specified by `pop_size`. For KLR, this
controls the max reasoning level `K`, e.g `pop_size=4` means `K=3` for KLR (includes
policies for `K=0`, `K=1`, `K=2`, and `K=3`) or `K=2` for KLR plus BR (KLR-BR) (includes
policies for `K=0`, `K=1`, `K=2`, and `BR`). For SP, this is the number of independent
policies in the population, including BR if using SP plus BR (SP-BR).

Training statistics are logged using tensorboard and wandb. To disable logging to wandb,
set `--track_wandb=False`. To disable all logging (i.e. for debugging), set
`--disable_logging=True`.

Use `--help` to see all available options.

"""
import argparse
from copy import deepcopy
from typing import Callable, Optional

import exp_utils
import posggym
from posggym.wrappers import FlattenObservations, RecordVideo

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.core import run_ppo
from posggym_baselines.ppo.ippo import IPPOConfig
from posggym_baselines.ppo.klr_ppo import KLRPPOConfig
from posggym_baselines.utils import strtobool


def get_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: Optional[int] = None
) -> Callable:
    """Get function for creating the environment."""

    def thunk():
        capture_video = config.capture_video and env_idx == 0 and worker_idx == 0
        render_mode = "rgb_array" if capture_video else None
        env = posggym.make(config.env_id, render_mode=render_mode, **config.env_kwargs)
        if capture_video:
            env = RecordVideo(env, config.video_dir)
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def train(args):
    """Run training for KLR and KLR plus BR."""
    env_data = exp_utils.get_env_data(None, None, full_env_id=args.full_env_id)
    env_kwargs = env_data.env_kwargs

    config_kwargs = deepcopy(exp_utils.DEFAULT_PPO_CONFIG)
    config_kwargs.update(
        {
            "exp_name": f"{args.pop_training_alg}_{args.full_env_id}_N{args.pop_size}",
            "env_creator_fn": get_env_creator_fn,
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    if args.pop_training_alg.startswith("KLR"):
        if args.pop_training_alg == "KLR-BR":
            include_BR = True
            max_reasoning_level = args.pop_size - 2
        else:
            include_BR = False
            max_reasoning_level = args.pop_size - 1

        config = KLRPPOConfig(
            max_reasoning_level=max_reasoning_level,
            include_BR=include_BR,
            filter_experience=True,
            **config_kwargs,
        )
    else:
        if args.pop_training_alg == "SP-BR":
            include_BR = True
            pop_size = args.pop_size - 1
        else:
            include_BR = False
            pop_size = args.pop_size

        config = IPPOConfig(
            pop_size=pop_size,
            include_BR=include_BR,
            filter_experience=True,
            **config_kwargs,
        )

    print(config)
    run_ppo(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pop_training_alg",
        type=str,
        choices=["SP", "SP-BR", "KLR", "KLR-BR"],
        help="The population training algorithm to run.",
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
        "--pop_size",
        type=int,
        default=4,
        help=(
            "Number of independent policies in the population. For `klr` and `klr-br`, "
            " this is the max reasoning level `K`, e.g `pop_size=4` means `K=3` for "
            "`klr` and `k=2` for `klr-br`."
        ),
    )
    parser.add_argument(
        "--track_wandb",
        type=strtobool,
        default=True,
        help="Whether to track the experiment with wandb.",
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
        default=int(3.2e6),
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
