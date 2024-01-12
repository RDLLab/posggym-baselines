import abc
from typing import Dict, Optional

import numpy as np
import posggym.model as M
import torch
from posggym.agents.policy import Policy, PolicyState
from posggym.agents.utils import processors
from posggym.agents.utils.action_distributions import DiscreteActionDistribution
from posggym.utils.history import AgentHistory
from torch.distributions.categorical import Categorical

from posggym_baselines.ppo.network import PPOLSTMModel


class SearchPolicy(abc.ABC):
    """A class for representing the search policy in MCTS planning."""

    def __init__(self, model: M.POSGModel, agent_id: str, policy_id: str):
        self.model = model
        self.agent_id = agent_id
        self.policy_id = policy_id

    @abc.abstractmethod
    def get_initial_state(self) -> PolicyState:
        """Get initial state of the policy.

        Returns
        -------
        initial_state : PolicyState
            the initial policy state

        """

    @abc.abstractmethod
    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        """Get the next policy state.

        Subclasses must implement this method.

        Arguments
        ---------
        action : ActType, optional
            the action performed. May be None if this is the first observation.
        obs : ObsType
            the observation received
        state : PolicyState
            the policy's state before action was performed and obs received

        Returns
        -------
        next_state : PolicyState
            the next policy state

        """

    @abc.abstractmethod
    def sample_action(self, state: PolicyState) -> M.ActType:
        """Sample an action given policy's current state.

        If the policy is deterministic then this will return the same action each time,
        given the same state. If the policy is stochastic then the action may change
        each time even if the state is the same.

        Subclasses must implement this method.

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        action : ActType
            the sampled action

        """

    @abc.abstractmethod
    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        """Get policy's distribution over actions for given policy state.

        Subclasses must implement this method

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        pi : Dict[M.ActType, float]
            the policy's distribution over actions

        """

    @abc.abstractmethod
    def get_value(self, state: PolicyState) -> float:
        """Get a value estimate of a history.

        Subclasses must implement this method, but may set it to raise a
        NotImplementedError if the policy does not support value estimates.

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        value : float
            the value estimate

        """

    def get_state_from_history(
        self, initial_state: PolicyState, history: AgentHistory
    ) -> PolicyState:
        """Get the policy's state given history.

        This function essentially unrolls the policy using the actions and observations
        contained in the agent history.

        Note, this function will return None for the action in the final output state,
        as this would correspond to the action that was selected by the policy to action

        Arguments
        ---------
        initial_state: PolicyState
            the policy's initial state
        history : AgentHistory
            the agent's action-observation history

        Returns
        -------
        state : PolicyState
            policy state given history

        """
        state = initial_state
        for a, o in history:
            state = self.get_next_state(a, o, state)
        return state

    def close(self):
        """Close policy and perform any necessary cleanup.

        Should be overridden in subclasses as necessary.
        """
        pass


class RandomSearchPolicy(SearchPolicy):
    """Uniform random search policy."""

    def __init__(self, model: M.POSGModel, agent_id: str):
        super().__init__(model, agent_id, "RandomSearchPolicy")
        self._action_space = model.action_spaces[agent_id]

    def get_initial_state(self) -> PolicyState:
        return {}

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> M.ActType:
        return self._action_space.sample()

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        return {a: 1.0 / self._action_space.n for a in range(self._action_space.n)}

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            "RandomSearchPolicy does not support value estimates."
        )


class SearchPolicyWrapper(SearchPolicy):
    """Wraps a posggym.agents Policy as a SearchPolicy."""

    def __init__(self, policy: Policy):
        super().__init__(policy.model, policy.agent_id, policy.policy_id)
        self.policy = policy
        self.action_space = list(range(policy.model.action_spaces[policy.agent_id].n))

    def get_initial_state(self) -> PolicyState:
        return self.policy.get_initial_state()

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        return self.policy.get_next_state(action, obs, state)

    def sample_action(self, state: PolicyState) -> M.ActType:
        return self.policy.sample_action(state)

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        pi = self.policy.get_pi(state).probs

        if len(pi) != len(self.action_space):
            for a in self.action_space:
                if a not in pi:
                    pi[a] = 0.0
        return pi

    def get_value(self, state: PolicyState) -> float:
        return self.policy.get_value(state)

    def close(self):
        self.policy.close()


def load_posggym_agents_search_policy(
    model: M.POSGModel, agent_id: str, policy_id: str
) -> "SearchPolicyWrapper":
    import posggym.agents as pga

    policy = pga.make(policy_id, model, agent_id)
    return SearchPolicyWrapper(policy)


class PPOLSTMSearchPolicy(SearchPolicy):
    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        policy_id: str,
        policy_model: PPOLSTMModel,
        obs_processor: processors.Processor,
    ):
        super().__init__(model, agent_id, policy_id)
        self.policy_model = policy_model
        self.action_space = list(range(model.action_spaces[agent_id].n))
        self.obs_processor = obs_processor

    def get_initial_state(self) -> PolicyState:
        lstm_shape = (self.policy_model.lstm_layers, 1, self.policy_model.lstm_size)
        lstm_state = (torch.zeros(lstm_shape).cpu(), torch.zeros(lstm_shape).cpu())
        return {"lstm_state": lstm_state, "pi": None, "value": 0.0}

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        obs = self.obs_processor(obs)
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            hidden_state, lstm_state = self.policy_model.get_states(
                obs, state["lstm_state"], done=torch.tensor([0])
            )
            logits = self.policy_model.actor(hidden_state)
            probs = Categorical(logits=logits).probs.squeeze()
            value = self.policy_model.critic(hidden_state)

        return {
            "lstm_state": lstm_state,
            "pi": DiscreteActionDistribution({a: probs[a] for a in self.action_space}),
            "value": value[0].item(),
        }

    def sample_action(self, state: PolicyState) -> M.ActType:
        return state["pi"].sample()

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        return state["pi"].probs

    def get_value(self, state: PolicyState) -> float:
        return state["value"]

    def close(self):
        return
