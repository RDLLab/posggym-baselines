"""K-Level Reasoning PPO (KLR-PPO) training for POSGGym grid world environments"""
import math
from pprint import pprint
import argparse
from typing import Callable, Optional
import yaml
from copy import deepcopy

import posggym
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.wrappers import FlattenObservations, RecordVideo

from posggym_baselines.utils import strtobool, NoOverwriteRecordVideo
from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.br_ppo import BRPPOConfig, UniformOtherAgentFn
from posggym_baselines.ppo.core import run_ppo, load_policies
from posggym_baselines.ppo.eval import (
    render_policies,
)


DEFAULT_CONFIG = {
    "eval_fns": [],
    # general config
    "exp_name": "br_ppo",
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
    "total_timesteps": int(1e8),
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

        env = AgentEnvWrapper(env, config.other_agent_fn)
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
        env = AgentEnvWrapper(env, config.other_agent_fn)
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
        env = AgentEnvWrapper(env, config.other_agent_fn)
        env = FlattenObservations(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


def load_config(args, env_fn):
    with open(args.env_kwargs_file, "r") as f:
        env_kwargs = yaml.safe_load(f)

    with open(args.other_agents_file, "r") as f:
        other_agent_policy_ids = yaml.safe_load(f)

    config_kwargs = deepcopy(DEFAULT_CONFIG)
    config_kwargs.update(
        {
            "env_creator_fn": env_fn,
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
        }
    )

    for k, v in vars(args).items():
        if k in config_kwargs:
            config_kwargs[k] = v

    config = BRPPOConfig(
        # BR-PPO specific config
        other_agent_ids=other_agent_policy_ids,
        other_agent_fn=UniformOtherAgentFn(other_agent_policy_ids),
        **config_kwargs,
    )
    pprint(config)
    return config


def train(args):
    config = load_config(args, get_env_creator_fn)
    run_ppo(config)


def render(args):
    config = load_config(args, get_render_env_creator_fn)
    policies = load_policies(config, args.saved_model_dir, device=config.eval_device)

    env = config.load_vec_env(num_envs=1)
    agent_ids = env.possible_agents

    for policy_id in policies:
        partner_dist = config.get_policy_partner_distribution(policy_id)
        render_policies(
            [{policy_id: policies[policy_id]}]
            + [
                {pi_id: policies[pi_id] for pi_id in partner_dist}
                for _ in agent_ids[1:]
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
        choices=["train", "render"],
        help="Whether to train new policies or evaluate some saved policies.",
    )
    parser.add_argument(
        "--env_kwargs_file",
        type=str,
        help="Path to YAML file containing env kwargs.",
    )
    parser.add_argument(
        "--other_agents_file",
        type=str,
        help="Path to YAML file containing other agent policy ids map.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="br_ppo",
        help="Name of experiment.",
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
        "--saved_model_dir",
        type=str,
        default=None,
        help="Directory containing saved models (required if running render).",
    )
    args = parser.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "render":
        assert args.saved_model_dir is not None
        render(args)
    else:
        raise ValueError(f"Invalid action {args.action}")
