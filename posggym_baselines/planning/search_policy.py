import abc
from typing import Dict, Optional

import posggym.model as M
from posggym.agents.policy import Policy
from posggym.utils.history import AgentHistory
from posggym.agents.policy import PolicyState


class SearchPolicy(abc.ABC):
    """A class for representing the search policy in MCTS planning."""

    def __init__(self, model: M.POSGModel, agent_id: str):
        self.model = model
        self.agent_id = agent_id

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
        super().__init__(model, agent_id)
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
        super().__init__(policy.model, policy.agent_id)
        self.policy = policy
        self.policy_id = policy.policy_id
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
