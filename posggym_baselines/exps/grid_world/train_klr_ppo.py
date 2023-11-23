"""K-Level Reasoning PPO (KLR-PPO) training for POSGGym grid world environments"""
import math
import argparse
from typing import Callable, Optional
import yaml
from copy import deepcopy

import posggym
from posggym.wrappers import FlattenObservations, RecordVideo
import seaborn as sns
import matplotlib.pyplot as plt

from posggym_baselines.utils import strtobool, NoOverwriteRecordVideo
from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.klr_ppo import KLRPPOConfig
from posggym_baselines.ppo.core import run_ppo, load_policies
from posggym_baselines.ppo.eval import (
    run_train_distribution_evaluation,
    run_all_pairwise_evaluation,
    render_policies,
)


DEFAULT_CONFIG = {
    "eval_fns": [],
    # general config
    "exp_name": "klr_ppo",
    "seed": 0,
    "cuda": True,
    "torch_deterministic": False,
    "worker_device": "cpu",
    "disable_logging": False,
    "track_wandb": True,
    "wandb_project": "posggym_baselines",
    "load_dir": None,
    "save_interval": -1,
    # env config
    "capture_video": False,
    # eval configuration
    "eval_interval": 100,
    "eval_episodes": 100,
    "num_eval_envs": 32,
    # training config
    "total_timesteps": int(3.2e7),
    "num_workers": 2,
    "num_envs": 32,
    "num_rollout_steps": 64,
    # batch size = 32 * 64 * 2 = 4096
    "seq_len": 10,
    "minibatch_size": 2048,
    # minibatch_num_seqs roughly = 2048 // 10 = 205
    "update_epochs": 2,
    # PPO update hyperparams
    "learning_rate": 3e-4,
    "anneal_lr": False,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 10,
    "target_kl": None,
    # model config
    "use_lstm": True,
    "lstm_size": 64,
    "lstm_num_layers": 1,
    "trunk_sizes": [64, 64],
    "head_sizes": [64],
}


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
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def get_render_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: Optional[int] = None
) -> Callable:
    """Get function for creating the environment with rendering."""

    def thunk():
        env = posggym.make(config.env_id, render_mode="human", **config.env_kwargs)
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def train(args):
    if args.env_kwargs_file is not None:
        with open(args.env_kwargs_file, "r") as f:
            env_kwargs = yaml.safe_load(f)
    else:
        env_kwargs = {}

    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": get_env_creator_fn,
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = KLRPPOConfig(
        # KLR-PPO specific config
        max_reasoning_level=args.max_reasoning_level,
        include_BR=args.include_BR,
        filter_experience=True,
        **config_kwargs,
    )
    print(config)

    run_ppo(config)


def eval(args):
    if args.env_kwargs_file is not None:
        with open(args.env_kwargs_file, "r") as f:
            env_kwargs = yaml.safe_load(f)
    else:
        env_kwargs = {}

    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": get_eval_env_creator_fn,
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
            "eval_episodes": 100,
        }
    )
    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = KLRPPOConfig(
        # KLR-PPO specific config
        max_reasoning_level=args.max_reasoning_level,
        include_BR=args.include_BR,
        filter_experience=True,
        **config_kwargs,
    )

    policies = load_policies(config, args.saved_model_dir, device=config.eval_device)

    if args.eval_train_distribution:
        pairwise_results = run_train_distribution_evaluation(policies, config)
        pairwise_returns = pairwise_results[0]
        pairwise_stds = pairwise_results[1]
    else:
        pairwise_results = run_all_pairwise_evaluation(policies, config)
        pairwise_returns = pairwise_results["pairwise_returns1"]
        pairwise_stds = pairwise_results["pairwise_returns_std1"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    sns.set_theme()
    sns.set_style("whitegrid")
    sns.heatmap(
        pairwise_returns,
        ax=axs[0],
        annot=True,
        fmt=".2f",
        cmap="viridis",
        yticklabels=list(policies),
        xticklabels=list(policies),
    )

    sns.heatmap(
        pairwise_stds,
        ax=axs[1],
        annot=True,
        fmt=".2f",
        cmap="viridis",
        yticklabels=list(policies),
        xticklabels=list(policies),
    )

    plt.show()


def render(args):
    if args.env_kwargs_file is not None:
        with open(args.env_kwargs_file, "r") as f:
            env_kwargs = yaml.safe_load(f)
    else:
        env_kwargs = {}

    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": get_render_env_creator_fn,
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
        }
    )
    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = KLRPPOConfig(
        # KLR-PPO specific config
        max_reasoning_level=args.max_reasoning_level,
        include_BR=args.include_BR,
        filter_experience=True,
        **config_kwargs,
    )

    policies = load_policies(config, args.saved_model_dir, device=config.eval_device)

    env = config.load_vec_env(num_envs=1)

    for policy_id in policies:
        partner_dist = config.get_policy_partner_distribution(policy_id)
        render_policies(
            [
                {policy_id: policies[policy_id]},
                {pi_id: policies[pi_id] for pi_id in partner_dist},
            ],
            num_episodes=25,
            env=env,
            config=config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "eval", "render"],
        help="Whether to train new policies or evaluate some saved policies.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        help="Grid world environmend ID.",
    )
    parser.add_argument(
        "--env_kwargs_file",
        type=str,
        default=None,
        help="Path to YAML file containing env kwargs.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="klr_ppo",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--max_reasoning_level",
        type=int,
        default=4,
        help="Number of reasoning levels K.",
    )
    parser.add_argument(
        "--include_BR",
        type=strtobool,
        default=True,
        help="Whether to include best-response policy in population.",
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
        "--saved_model_dir",
        type=str,
        default=None,
        help="Directory containing saved models (required if running eval).",
    )
    parser.add_argument(
        "--eval_train_distribution",
        type=strtobool,
        default=True,
        help="Whether to run evaluation for training distribution only.",
    )
    args = parser.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "eval":
        assert args.saved_model_dir is not None
        eval(args)
    elif args.action == "render":
        assert args.saved_model_dir is not None
        render(args)
    else:
        raise ValueError(f"Invalid action {args.action}")
