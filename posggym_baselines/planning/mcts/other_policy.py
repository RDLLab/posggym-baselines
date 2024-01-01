from __future__ import annotations

import abc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import posggym.model as M
    from posggym.utils.history import AgentHistory
    from posggym.agents.policy import PolicyState


class OtherAgentPolicy(abc.ABC):
    """A class for representing the policy of the other agent in the MCTS planning."""

    def __init__(self, model: M.POSGModel, agent_id: str, policy_id: str):
        self.model = model
        self.agent_id = agent_id
        self.policy_id = policy_id

    @abc.abstractmethod
    def sample_initial_state(self) -> PolicyState:
        """Get an initial state of the policy.

        Returns
        -------
        initial_state : PolicyState
            the initial policy state

        """

    @abc.abstractmethod
    def get_next_state(
        self,
        action: M.ActType | None,
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


class RandomOtherAgentPolicy(OtherAgentPolicy):
    """Uniform random policy."""

    def __init__(self, model: M.POSGModel, agent_id: str, policy_id: str):
        super().__init__(model, agent_id, policy_id)
        self._action_space = model.action_spaces[agent_id]

    def sample_initial_state(self) -> PolicyState:
        return {}

    def get_next_state(
        self,
        action: M.ActType | None,
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> M.ActType:
        return self._action_space.sample()
