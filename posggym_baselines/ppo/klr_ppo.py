"""K-Level Reasoning PPO (KLR-PPO) population.

This module implements the K-Level Reasoning PPO (KLR-PPO) population training, which is
a population of K-Level Reasoning policies where level 0 is a best-response policy to
the uniform random policy, and each level k > 0 is a best-response policy to the level
k-1 policy.
"""
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.network import PPOLSTMModel, PPOMLPModel, PPOModel


class UniformRandomModel(PPOModel):
    """Unirform random policy implementing PPOModel interface."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.action_probs = torch.ones((1, num_actions)) / num_actions
        self.action_probs.requires_grad = False

    def get_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
    ) -> torch.tensor:
        return torch.zeros((x.shape[0], 1))

    def get_action(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        action_probs = self.action_probs.repeat(x.shape[0], 1)
        probs = Categorical(probs=action_probs)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            lstm_state,
        )

    def get_action_and_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        action_probs = self.action_probs.repeat(x.shape[0], 1)
        probs = Categorical(probs=action_probs)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.get_value(x, lstm_state, done),
            lstm_state,
        )


@dataclass
class KLRPPOConfig(PPOConfig):
    # name of the experiment
    exp_name: str = "klr_ppo"
    # number of reasoning levels K
    max_reasoning_level: int = 4
    # whether to train best-response policy on top of population
    include_BR: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.max_reasoning_level >= 0
        self.filter_experience = True

        env = self.env_creator_fn(self, 0, None)()
        self.num_agents = len(env.possible_agents)

    def load_policies(self, device: Optional[torch.device]) -> Dict[str, PPOModel]:
        if self.use_lstm:
            model_cls = PPOLSTMModel
            model_kwargs = {
                "input_size": np.prod(self.obs_space.shape),
                "num_actions": self.act_space.n,
                "trunk_sizes": self.trunk_sizes,
                "lstm_size": self.lstm_size,
                "lstm_layers": self.lstm_num_layers,
                "head_sizes": self.head_sizes,
            }
        else:
            model_cls = PPOMLPModel
            model_kwargs = {
                "input_size": np.prod(self.obs_space.shape),
                "num_actions": self.act_space.n,
                "trunk_sizes": self.trunk_sizes,
                "head_sizes": self.head_sizes,
            }

        policies = {
            f"k_{i}": model_cls(**model_kwargs).to(device)
            for i in range(self.max_reasoning_level + 1)
        }
        if self.include_BR:
            policies["BR"] = model_cls(**model_kwargs).to(device)
        policies["random"] = UniformRandomModel(self.act_space.n).to(device)
        return policies

    def get_policy_partner_distribution(self, policy_id: str) -> Dict[str, float]:
        if policy_id == "BR":
            klr_policy_ids = self.get_klr_policy_ids()
            return {
                partner_id: 1.0 / len(klr_policy_ids) for partner_id in klr_policy_ids
            }
        # note that we don't include BR in the partner distribution for KLR
        # since episodes against BR are not used for training SP
        if policy_id == "random":
            return {}
        k = int(policy_id.split("_")[-1])
        if k == 0:
            return {"random": 1.0}
        return {f"k_{k-1}": 1.0}

    def sample_episode_policies(self) -> List[str]:
        # -1 to exclude "random"
        policy_id1 = random.choice(self.get_all_policy_ids()[:-1])
        if policy_id1 == "BR":
            policy_id2 = random.choice(self.get_klr_policy_ids())
        elif policy_id1 == "k_0":
            policy_id2 = "random"
        else:
            k = int(policy_id1.split("_")[-1])
            policy_id2 = f"k_{k-1}"

        policy_ids = [policy_id1] + [policy_id2] * (self.num_agents - 1)
        random.shuffle(policy_ids)
        return policy_ids

    def get_all_policy_ids(self) -> List[str]:
        klr_policy_ids = self.get_klr_policy_ids()
        if self.include_BR:
            klr_policy_ids.append("BR")
        klr_policy_ids.append("random")
        return klr_policy_ids

    def get_klr_policy_ids(self) -> List[str]:
        return [f"k_{i}" for i in range(self.max_reasoning_level + 1)]

    @property
    def train_policies(self) -> List[str]:
        # -1 to exclude "random"
        return self.get_all_policy_ids()[:-1]
