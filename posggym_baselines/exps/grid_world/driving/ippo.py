"""Independent PPO (IPPO) training for POSGGym grid world environments"""
import argparse
from typing import Callable, Optional
import yaml

import posggym
from posggym.wrappers import FlattenObservations, RecordVideo

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.ippo import IPPOConfig
from posggym_baselines.ppo.core import run_ppo


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1';
    false values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


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


def main(args):
    if args.env_kwargs_file is not None:
        with open(args.env_kwargs_file, "r") as f:
            env_kwargs = yaml.safe_load(f)
    else:
        env_kwargs = {}

    config_kwargs = {
        "env_creator_fn": get_env_creator_fn,
        "eval_fns": [],
        # general config
        "exp_name": "ippo",
        "seed": 0,
        "cuda": True,
        "torch_deterministic": False,
        "worker_device": "cpu",
        "disable_logging": False,
        "track_wandb": True,
        "wandb_project": "posggym_baselines",
        "load_dir": None,
        "save_interval": -1,
        # eval configuration
        "eval_interval": 100,
        "eval_episodes": 100,
        "num_eval_envs": 32,
        # env config
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "capture_video": args.capture_video,
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
        "trunk_size": [64, 64],
        "head_size": [64],
    }

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = IPPOConfig(
        # IPPO specific config
        pop_size=args.pop_size,
        include_BR=args.include_BR,
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
        default="ippo",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=4,
        help="Number of independent policies in the population.",
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
        default=True,
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
    main(parser.parse_args())
