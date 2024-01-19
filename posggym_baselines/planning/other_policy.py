import abc
import random
from typing import Dict, List, Optional

import posggym.model as M
from posggym.agents.policy import Policy, PolicyState
from posggym.utils.history import AgentHistory


class OtherAgentPolicy(abc.ABC):
    """A class for representing the policy of the other agent in the MCTS planning."""

    def __init__(self, model: M.POSGModel, agent_id: str):
        self.model = model
        self.agent_id = agent_id

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

    def __init__(self, model: M.POSGModel, agent_id: str):
        super().__init__(model, agent_id)
        self._action_space = model.action_spaces[agent_id]

    def sample_initial_state(self) -> PolicyState:
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


class OtherAgentMixturePolicy(OtherAgentPolicy):
    """Other agent mixture policy.

    This policy is a distribution over a set of possible policies for the other agent.
    Only one of the policies is active at a time (i.e. it's consistent across an
    episode). The active policy is sampled in the `sample_initial_state` method.

    Currently assumes uniform prior over policies and only 2 agents in the environment
    (i.e. the planning agent and the other agent)
    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        policies: Dict[str, Policy],
    ):
        super().__init__(model, agent_id)
        assert len(model.possible_agents) == 2, "Currently only supports 2 agents"
        self.policies = policies
        self.action_space = list(range(model.action_spaces[agent_id].n))

    def sample_initial_state(self) -> PolicyState:
        policy_id = random.choice(list(self.policies))
        return {
            "policy_id": policy_id,
            "policy_state": self.policies[policy_id].get_initial_state(),
        }

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        policy_id = state["policy_id"]
        policy_state = state["policy_state"]
        next_policy_state = self.policies[policy_id].get_next_state(
            action, obs, policy_state
        )
        return {"policy_id": policy_id, "policy_state": next_policy_state}

    def sample_action(self, state: PolicyState) -> M.ActType:
        policy_id = state["policy_id"]
        policy_state = state["policy_state"]
        return self.policies[policy_id].sample_action(policy_state)

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        policy_id = state["policy_id"]
        policy_state = state["policy_state"]
        pi = self.policies[policy_id].get_pi(policy_state).probs

        if len(pi) != len(self.action_space):
            for a in self.action_space:
                if a not in pi:
                    pi[a] = 0.0
        return pi

    def close(self):
        for policy in self.policies.values():
            policy.close()

    @staticmethod
    def load_posggym_agents_policy(
        model: M.POSGModel, agent_id: str, policy_ids: List[str]
    ) -> "OtherAgentMixturePolicy":
        import posggym.agents as pga

        policies = {}
        for policy_id in policy_ids:
            policies[policy_id] = pga.make(policy_id, model, agent_id)

        return OtherAgentMixturePolicy(model, agent_id, policies)
