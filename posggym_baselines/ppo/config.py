"""Base configuration for RCPD."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import torch
from posggym.vector import SyncVectorEnv
from posggym.wrappers import RecordEpisodeStatistics, StackEnv

from posggym_baselines.config import BASE_RESULTS_DIR
from posggym_baselines.ppo.eval import EvalFn, run_all_pairwise_evaluation


if TYPE_CHECKING:
    from pathlib import Path

    import gymnasium as gym
    from posggym.agents.policy import Policy
    from posggym.agents.utils.processors import Processor

    from posggym_baselines.ppo.network import PPOModel


@dataclass
class PPOConfig:
    """PPO configuration.

    Contains parameters that are common to all experiments.
    """

    # function for getting the environment creator function
    # Callable[, posggym.Env]
    env_creator_fn: callable

    # List of evaluation functions to use
    eval_fns: List[EvalFn] = field(
        default_factory=lambda: [run_all_pairwise_evaluation]
    )

    # The name of this experiment
    exp_name: str = "ppo"
    # The name of this run
    # run_name = f"{exp_name}_{env_id}_{seed}_{time}"
    run_name: str = field(init=False)
    # Experiment seed
    seed: int = 0
    # Whether to use CUDA
    cuda: bool = True
    # Whether to set torch to deterministic mode
    # `torch.backends.cudnn.deterministic=False`
    torch_deterministic: bool = True
    # Device to use for learner model
    device: torch.device = field(init=False)
    # Device to use for running rollout workers
    worker_device: torch.device = torch.device("cpu")
    # whether to disable logging (for testing purposes)
    disable_logging: bool = False
    # Whether to log to wandb or not
    track_wandb: bool = False
    # wandb project name
    wandb_project: str = "posggym_dynamics"
    # wandb entity name
    wandb_entity: str = None
    # wandb group name
    wandb_group: str = None
    # Directory where the model and logs will be saved
    log_dir: Path = None
    # Directory where videos will be saved
    video_dir: Path = field(init=False)
    # Directory where models will be saved
    model_dir: Path = field(init=False)
    # optional directory to load existing algorithm/model from
    load_dir: Path = None
    # number of updates (i.e. batches) after which the model/algorithm is saved
    # if 0, never save
    # if > 0, save every save_interval updates
    # if -1, save only at the end of training
    save_interval: int = -1

    # number of updates between evaluations against heldout policies, if applicable
    # if 0 = no evaluation is performed,
    # >0 = evaluation is performed every eval_interval updates,
    # -1 = evaluation is performed at the end of training
    eval_interval: int = 0
    # number of episodes to evaluate against heldout policies
    eval_episodes: int = 100
    # number of parallel environments to use for evaluation
    num_eval_envs: int = 32
    # device to use for evaluation
    eval_device: torch.device = torch.device("cpu")

    # ID of environment
    env_id: str = "rps-v0"
    # keyword arguments for the environment creator function
    env_kwargs: dict = field(default_factory=dict)
    # whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False
    # observation space of the environment
    obs_space: gym.spaces.Box = field(init=False)
    # action space of the environment
    act_space: gym.spaces.Discrete | gym.spaces.MultiDiscrete = field(init=False)
    # number of agents in the environment
    num_agents: int = field(init=False)

    # total timesteps for training
    total_timesteps: int = 10000000
    # total number of updates during training
    num_updates: int = field(init=False)
    # number of steps per update batch
    # *warning*: needs to be initialized in post_init of child class
    batch_size: int = field(init=False)
    # number of parallel workers to use
    num_workers: int = 1
    # number of parallel environments per worker to use for collecting trajectories
    num_envs: int = 16
    # number of steps per rollout
    num_rollout_steps: int = 128
    # number of steps per sequence chunk for BPTT
    seq_len: int = 16
    # Number of steps in each mini-batch.
    minibatch_size: int = 2048
    # Number of sequence chunks in each mini-batch.
    # minibatch_num_seqs = minibatch_size // seq_len
    minibatch_num_seqs: int = field(init=False)
    # Number of epochs to train policy per update
    update_epochs: int = 2
    # whether to filter experience for each policy by the policy's partner distribution
    filter_experience: bool = True

    # Learning rate of the optimizer
    learning_rate: float = 2.5e-4
    # Whether to anneal the learning rate linearly to zero
    anneal_lr: bool = True
    # The discount factor
    gamma: float = 0.99
    # The GAE lambda parameter
    gae_lambda: float = 0.95
    # Whether to normalize advantages
    norm_adv: bool = True
    # Surrogate clip coefficient of PPO
    clip_coef: float = 0.2
    # Whether to use a clipped loss for the value function, as per the paper
    clip_vloss: bool = True
    # Coefficient of the value function loss
    vf_coef: float = 0.5
    # Coefficient of the entropy
    ent_coef: float = 0.01
    # The maximum norm for the gradient clipping
    max_grad_norm: float = 0.5
    # The target KL divergence threshold
    target_kl: Optional[float] = None

    # Whether to use a recurrent policy
    use_lstm: bool = True
    # Number of hidden units in the LSTM
    lstm_size: int = 64
    # Number of layers in the LSTM
    lstm_num_layers: int = 1
    # size of network trunk
    trunk_sizes: List[int] = field(default_factory=lambda: [64])
    # size of network heads
    head_sizes: List[int] = field(default_factory=lambda: [64])

    def __post_init__(self):
        """Post initialization."""
        self.run_name = self.exp_name
        if self.env_id not in self.exp_name:
            self.run_name += f"_{self.env_id}"
        self.run_name += f"_{self.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.log_dir = (
            self.log_dir if self.log_dir is not None else BASE_RESULTS_DIR
        ) / self.run_name
        self.video_dir = self.log_dir / "videos"
        self.model_dir = self.log_dir / "models"

        if not self.disable_logging:
            self.log_dir.parent.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            self.video_dir.mkdir(exist_ok=True)
            self.model_dir.mkdir(exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )
        self.worker_device = torch.device(self.worker_device if self.cuda else "cpu")
        self.eval_device = torch.device(self.eval_device if self.cuda else "cpu")

        self.batch_size = self.num_rollout_steps * self.num_envs * self.num_workers
        self.minibatch_num_seqs = self.minibatch_size // self.seq_len

        self.num_updates = (self.total_timesteps // self.batch_size) + int(
            self.total_timesteps % self.batch_size != 0
        )

        if self.save_interval == -1:
            self.save_interval = self.num_updates

        if len(self.eval_fns) == 0:
            self.eval_interval = 0
        elif self.eval_interval == -1:
            self.eval_interval = self.num_updates

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = self.torch_deterministic

        env = self.env_creator_fn(self, 0, None)()
        self.num_agents = len(env.possible_agents)

        self.obs_space = env.observation_spaces[env.possible_agents[0]]
        for i, obs_space in env.observation_spaces.items():
            assert obs_space == self.obs_space, "All agents must have same obs space"

        self.act_space = env.action_spaces[env.possible_agents[0]]
        for i, act_space in env.action_spaces.items():
            assert act_space == self.act_space, "All agents must have same act space"

    def load_vec_env(self, num_envs: int | None = None, worker_idx: int | None = None):
        """Load the vectorized environment."""
        if num_envs is None:
            num_envs = self.num_envs
        env = SyncVectorEnv(
            [self.env_creator_fn(self, i, worker_idx) for i in range(num_envs)]
        )
        env = RecordEpisodeStatistics(env)
        env = StackEnv(env)
        return env

    @property
    def train_policies(self) -> List[str]:
        """IDs of policies that are being trained using PPO."""
        raise NotImplementedError

    def load_policies(
        self, device: torch.device | None
    ) -> Dict[str, Union[PPOModel, Policy]]:
        """Load models of all policies for algorithm."""
        raise NotImplementedError

    def get_obs_processors(self) -> Dict[str, Processor]:
        """Get the observation processors for each policy."""
        raise NotImplementedError

    def sample_episode_policies(self) -> List[str]:
        """Sample policies to use for each agent in an episode.

        Returns
        -------
        List[str]
            List of policy_id's one for each agent in the environment.
        """
        raise NotImplementedError

    def get_policy_partner_distribution(self, policy_id: str) -> Dict[str, float]:
        """Get the distribution of partner policies for the given policy ID.

        Arguments
        ---------
        policy_id : str
            The ID of the policy to get distribution for.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping from policy ID to probability.
        """
        raise NotImplementedError

    def get_all_policy_ids(self) -> List[str]:
        """Get the IDs of all policies."""
        raise NotImplementedError

    def get_policy_idx_to_id_mapping(self) -> Dict[int, str]:
        """Get the mapping from policy index to policy ID."""
        return dict(enumerate(self.get_all_policy_ids()))

    def get_policy_id_to_idx_mapping(self) -> Dict[str, int]:
        """Get the mapping from policy ID to policy index."""
        return {v: k for k, v in self.get_policy_idx_to_id_mapping().items()}

    def get_policy_idx(self, policy_id: str) -> int:
        """Get the index of the policy."""
        return self.get_policy_id_to_idx_mapping()[policy_id]

    def get_policy_id(self, policy_idx: int) -> str:
        """Get the ID of the policy."""
        return self.get_policy_idx_to_id_mapping()[policy_idx]

    def asdict(self) -> Dict:
        return asdict(self)

    def aspickleable(self) -> Dict:
        """Get a pickleable version of the config."""
        return {
            k: v
            for k, v in self.asdict().items()
            if k not in ("env_creator_fn", "eval_fns")
        }
