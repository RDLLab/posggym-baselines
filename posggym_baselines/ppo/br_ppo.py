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


class UniformOtherAgentFn:
    """Function for loading other agents.

    Samples the other agents from a uniform distribution over set of possible agents.

    This is a callable class that can be pickled and passed to the workers.
    """

    def __init__(self, agent_policy_ids: Dict[str, List[str]]):
        self.agent_policy_ids = agent_policy_ids
        self.policies = {i: {} for i in agent_policy_ids}

    def __call__(self, model: posggym.POSGModel) -> Dict[str, pga.Policy]:
        other_agents = {}
        for agent_id in self.agent_policy_ids:
            pi_id = model.rng.choice(self.agent_policy_ids[agent_id])
            if pi_id not in self.policies[agent_id]:
                self.policies[agent_id][pi_id] = pga.make(pi_id, model, agent_id)
            other_agents[agent_id] = self.policies[agent_id][pi_id]
        return other_agents


@dataclass
class BRPPOConfig(PPOConfig):
    # name of the experiment
    exp_name: str = "br_ppo"
    # other agent policy ids
    other_agent_ids: Dict[str, List[str]] = None
    # function for generating other agents
    other_agent_fn: Callable[[posggym.POSGModel], Dict[str, pga.Policy]] = None

    def __post_init__(self):
        if self.other_agent_fn is None:
            assert self.other_agent_ids is not None
            self.other_agent_fn = UniformOtherAgentFn(self.other_agent_ids)
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
