import random
from typing import Optional, Dict, List

import posggym.model as M
from posggym.agents.policy import PolicyState, Policy


from posggym_baselines.planning.mcts.other_policy import OtherAgentPolicy


class POTMMCPOtherAgentPolicy(OtherAgentPolicy):
    """Other agent policy for the POTMMCP algorithm.

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

    def close(self):
        for policy in self.policies.values():
            policy.close()

    @staticmethod
    def load_posggym_agents_policy(
        model: M.POSGModel, agent_id: str, policy_ids: List[str]
    ) -> "POTMMCPOtherAgentPolicy":
        import posggym.agents as pga

        policies = {}
        for policy_id in policy_ids:
            policies[policy_id] = pga.make(policy_id, model, agent_id)

        return POTMMCPOtherAgentPolicy(model, agent_id, policies)
