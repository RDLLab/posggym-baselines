"""Best Responst PPO (BR-PPO).

This module implements the Best-Response PPO (BR-PPO) training which trains a single
PPO policy against a distribution of fixed other agents.

It assumes the environment is a POSGGym environment that is wrapped using the 
`posggym.agents.wrappers.AgentEnvWrapper` wrapper, so that the fixed other agent 
distribution is part of the environment.

"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable

import posggym
import posggym.agents as pga
import numpy as np
import torch

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.network import PPOModel, PPOLSTMModel


@dataclass
class BRPPOConfig(PPOConfig):
    # name of the experiment
    exp_name: str = "br_ppo"
    # function for generating other agents
    other_agent_fn: Callable[[posggym.POSGModel], Dict[str, pga.Policy]] = None

    def __post_init__(self):
        if self.other_agent_fn is None:
            raise ValueError("Must provide other_agent_fn.")
        super().__post_init__()

        env = self.env_creator_fn(self, 0, None)()
        self.num_agents = len(env.possible_agents)

    def load_policies(self, device: Optional[torch.device]) -> Dict[str, PPOModel]:
        policies = {
            "BR": PPOLSTMModel(
                input_size=np.prod(self.obs_space.shape),
                num_actions=self.act_space.n,
                trunk_sizes=self.trunk_sizes,
                lstm_size=self.lstm_size,
                lstm_layers=self.lstm_num_layers,
                head_sizes=self.head_sizes,
            ).to(device)
        }
        return policies

    def get_policy_partner_distribution(self, policy_id: str) -> Dict[str, float]:
        return {"BR": 1.0}

    def sample_episode_policies(self) -> List[str]:
        return ["BR"] * self.num_agents

    def get_all_policy_ids(self) -> List[str]:
        return ["BR"]

    @property
    def train_policies(self) -> List[str]:
        return self.get_all_policy_ids()
