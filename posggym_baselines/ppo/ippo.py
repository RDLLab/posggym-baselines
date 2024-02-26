"""Independent PPO (IPPO) population.

This module implements the independent PPO (IPPO) population, which is a population of
independent PPO policies, where each policy is trained in self-play.
"""
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.network import PPOLSTMModel, PPOMLPModel, PPOModel
from gymnasium import spaces


@dataclass
class IPPOConfig(PPOConfig):
    # name of the experiment
    exp_name: str = "ippo"
    # number of independent policies in the population (not including BR)
    pop_size: int = 4
    # whether to train best-response policy on top of population
    include_BR: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.pop_size > 0
        if self.include_BR:
            self.filter_experience = True

        env = self.env_creator_fn(self, 0, None)()
        self.num_agents = len(env.possible_agents)

    def load_policies(self, device: Optional[torch.device]) -> Dict[str, PPOModel]:
        num_actions = (
            self.act_space.n
            if isinstance(self.act_space, spaces.Discrete)
            else self.act_space.nvec.tolist()
        )
        one_hot_size = (
            self.act_space.n
            if isinstance(self.act_space, spaces.Discrete)
            else self.act_space.nvec.sum()
        )
        if self.use_lstm:
            model_cls = PPOLSTMModel
            model_kwargs = {
                "input_size": np.prod(self.obs_space.shape) + one_hot_size
                if self.use_previous_action
                else 0,
                "num_actions": num_actions,
                "trunk_sizes": self.trunk_sizes,
                "lstm_size": self.lstm_size,
                "lstm_layers": self.lstm_num_layers,
                "head_sizes": self.head_sizes,
            }
        else:
            model_cls = PPOMLPModel
            model_kwargs = {
                "input_size": np.prod(self.obs_space.shape),
                "num_actions": num_actions,
                "trunk_sizes": self.trunk_sizes,
                "head_sizes": self.head_sizes,
            }

        policies = {
            f"sp_{i}": model_cls(**model_kwargs).to(device)
            for i in range(self.pop_size)
        }
        if self.include_BR:
            policies["BR"] = model_cls(**model_kwargs).to(device)
        return policies

    def get_policy_partner_distribution(self, policy_id: str) -> Dict[str, float]:
        if policy_id == "BR":
            sp_policy_ids = self.get_sp_policy_ids()
            return {
                partner_id: 1.0 / len(sp_policy_ids) for partner_id in sp_policy_ids
            }
        # note that we don't include BR in the partner distribution for SP
        # since episodes against BR are not used for training SP
        return {policy_id: 1.0}

    def sample_episode_policies(self) -> List[str]:
        idx1 = torch.randint(self.pop_size + int(self.include_BR), size=(1,))[0]
        if idx1 == self.pop_size:
            policy_id2 = random.choice(self.get_sp_policy_ids())
            policy_ids = ["BR"] + [policy_id2] * (self.num_agents - 1)
            random.shuffle(policy_ids)
        else:
            policy_ids = [f"sp_{idx1}"] * self.num_agents

        return policy_ids

    def get_all_policy_ids(self) -> List[str]:
        sp_policy_ids = self.get_sp_policy_ids()
        if self.include_BR:
            return sp_policy_ids + ["BR"]
        return sp_policy_ids

    def get_sp_policy_ids(self) -> List[str]:
        return [f"sp_{i}" for i in range(self.pop_size)]

    @property
    def train_policies(self) -> List[str]:
        return self.get_all_policy_ids()
