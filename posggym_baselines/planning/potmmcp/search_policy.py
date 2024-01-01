import math
import random
from typing import Optional, Dict

import posggym.model as M
from posggym.agents.policy import PolicyState, Policy


from posggym_baselines.planning.mcts.search_policy import (
    SearchPolicy,
    SearchPolicyWrapper,
)


class POTMMCPMetaPolicy(SearchPolicy):
    """POTMMCP Meta-Policy used for search policy for the POTMMCP algorithm.

    This policy maps from the other agent policy ID to a distribution over a set of
    possible policies for the planning agent: the `meta_policy`. The meta policy is then
    used to generate the distribution over actions.

    Currently assumes only 2 agents in the environment(i.e. the planning agent and the
    other agent) and uniform prior over other agent policies.

    Within the POTMMCP algorithm, this policy is used as the search policy, but in
    a non-standard way. Specifically:

    1. when POTMMCP runs simulations, for each simulation the meta-policy is used to
    sample a policy from the set of policies that is then used as the search policy
    (this is done via the `sample_policy` method)
        - the sampled policy is used to generate the initial action probabilities
          for the leaf node added to the search tree after each simulation and for leaf
          node evaluations (either using value function or rollout)
    2. at the start of planning, the meta-policy is used to compute the action
    probabilities of the root node (this is done via the `get_expected_action_probs`)
    assuming a uniform prior over other agent policies

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        policies: Dict[str, Policy],
        meta_policy: Dict[str, Dict[str, float]],
    ):
        super().__init__(model, agent_id)
        assert len(model.possible_agents) == 2, "Currently only supports 2 agents"
        assert len(meta_policy) > 0
        for meta_policy_dist in meta_policy.values():
            assert all(k in policies for k in meta_policy_dist)
            assert abs(sum(meta_policy_dist.values()) - 1) < 1e-6

        self.other_agent_id = [i for i in model.possible_agents if i != agent_id][0]
        self.policies = policies
        self.meta_policy = meta_policy
        self.action_space = list(range(model.action_spaces[agent_id].n))

    def get_initial_state(self) -> PolicyState:
        return {pi_id: pi.get_initial_state() for pi_id, pi in self.policies.items()}

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        return {
            pi_id: self.policies[pi_id].get_next_state(action, obs, pi_state)
            for pi_id, pi_state in state.items()
        }

    def sample_action(self, state: PolicyState) -> M.ActType:
        raise NotImplementedError(
            "POTMMCPMetaPolicy does not support action sampling. Instead, use "
            "sample_policy to sample a policy and then sample an action from that."
        )

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        raise NotImplementedError(
            "POTMMCPMetaPolicy does not support action sampling. Instead, use "
            "sample_policy to sample a policy and then get the action distribution from "
            "that."
        )

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            "POTMMCPMetaPolicy does not support value estimates. Instead, use "
            "sample_policy to sample a policy and then get the value from that."
        )

    def sample_policy(self, other_agent_policy_state: Dict[str, PolicyState]) -> Policy:
        """Sample policy to use as search policy given policies of other agents."""
        meta_policy_dist = self.meta_policy[
            other_agent_policy_state[self.other_agent_id]["policy_id"]
        ]
        policy_id = random.choices(
            list(meta_policy_dist), weights=list(meta_policy_dist.values()), k=1
        )[0]
        return SearchPolicyWrapper(self.policies[policy_id])

    def get_expected_action_probs(
        self,
        other_agent_policy_dist: Optional[Dict[str, float]],
        policy_state: PolicyState,
    ) -> Dict[M.ActType, float]:
        """Get action probabilities from distribution over other agent policies.

        If `other_agent_policy_dist` is None, then assume uniform prior over other
        agent policies.

        `policy_state` is the state of the meta-policy (i.e. the state of each ego
        policy in the meta-policy).

        """
        if other_agent_policy_dist is None:
            # assume uniform prior over other agent policies
            other_agent_policy_dist = {
                pi_id: 1.0 / len(self.meta_policy) for pi_id in self.meta_policy
            }

        # get distribution over ego policies
        expected_meta_policy = {pi_id: 0.0 for pi_id in self.policies}
        for other_agent_policy_id, prob in other_agent_policy_dist.items():
            meta_policy_dist = self.meta_policy[other_agent_policy_id]
            for pi_id, meta_prob in meta_policy_dist.items():
                expected_meta_policy[pi_id] += prob * meta_prob

        # normalize
        dist_sum = sum(expected_meta_policy.values())
        for pi_id in expected_meta_policy:
            expected_meta_policy[pi_id] /= dist_sum

        # get distribution over actions
        expected_action_dist = {a: 0.0 for a in self.action_space}
        for pi_id, pi_prob in expected_meta_policy.items():
            pi, pi_state = self.policies[pi_id], policy_state[pi_id]
            for a, a_prob in pi.get_pi(pi_state).probs.items():
                expected_action_dist[a] += pi_prob * a_prob

        # normalize
        prob_sum = sum(expected_action_dist.values())
        for a in expected_action_dist:
            expected_action_dist[a] /= prob_sum

        return expected_action_dist

    @staticmethod
    def load_posggym_agents_meta_policy(
        model: M.POSGModel, agent_id: str, meta_policy: Dict[str, Dict[str, float]]
    ) -> "POTMMCPMetaPolicy":
        import posggym.agents as pga

        policy_ids = set()
        for policy_dist in meta_policy.values():
            policy_ids.update(policy_dist)

        policies = {}
        for policy_id in policy_ids:
            policies[policy_id] = pga.make(policy_id, model, agent_id)

        return POTMMCPMetaPolicy(model, agent_id, policies, meta_policy)

    @staticmethod
    def get_uniform_meta_policy(
        pairwise_returns: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Get uniform meta-policy from pairwise returns.

        `pairwise_returns` is a dictionary mapping from ego agent policy ID to a
        dictionary mapping from other agent policy ID to the return of the ego agent
        policy against that other agent policy.

        """
        other_agent_pi_ids = set()
        for returns_map in pairwise_returns.values():
            other_agent_pi_ids.update(returns_map)

        meta_policy = {
            other_agent_pi_id: {
                pi_id_i: 1.0 / len(pairwise_returns) for pi_id_i in pairwise_returns
            }
            for other_agent_pi_id in other_agent_pi_ids
        }
        return meta_policy

    @staticmethod
    def get_greedy_meta_policy(
        pairwise_returns: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Get greedy meta-policy from pairwise returns.

        `pairwise_returns` is a dictionary mapping from ego agent policy ID to a
        dictionary mapping from other agent policy ID to the return of the ego agent
        policy against that other agent policy.

        """
        other_agent_pi_ids = set()
        for returns_map in pairwise_returns.values():
            other_agent_pi_ids.update(returns_map)

        meta_policy = {
            other_agent_pi_id: {} for other_agent_pi_id in other_agent_pi_ids
        }
        for other_agent_pi_id in other_agent_pi_ids:
            max_return = -float("inf")
            max_policies = []
            for ego_agent_pi_id, returns_map in pairwise_returns.items():
                if returns_map[other_agent_pi_id] > max_return:
                    max_return = returns_map[other_agent_pi_id]
                    max_policies = [ego_agent_pi_id]
                elif returns_map[other_agent_pi_id] == max_return:
                    max_policies.append(ego_agent_pi_id)

            for ego_agent_pi_id in pairwise_returns:
                if ego_agent_pi_id in max_policies:
                    prob = 1.0 / len(max_policies)
                else:
                    prob = 0.0
                meta_policy[other_agent_pi_id][ego_agent_pi_id] = prob

        return meta_policy

    @staticmethod
    def get_softmax_meta_policy(
        pairwise_returns: Dict[str, Dict[str, float]], temperature: float
    ) -> Dict[str, Dict[str, float]]:
        """Get softmax meta-policy from pairwise returns.

        `pairwise_returns` is a dictionary mapping from ego agent policy ID to a
        dictionary mapping from other agent policy ID to the return of the ego agent
        policy against that other agent policy.

        `temperature` is the softmax temperature.

        """
        other_agent_pi_ids = set()
        for returns_map in pairwise_returns.values():
            other_agent_pi_ids.update(returns_map)

        meta_policy = {
            other_agent_pi_id: {} for other_agent_pi_id in other_agent_pi_ids
        }
        for other_agent_pi_id in other_agent_pi_ids:
            returns = [
                pairwise_returns[pi_id][other_agent_pi_id] for pi_id in pairwise_returns
            ]
            sum_e = sum(math.e ** (z / temperature) for z in returns)
            softmax_probs = [math.e ** (z / temperature) / sum_e for z in returns]

            meta_policy[other_agent_pi_id] = {
                pi_id: softmax_probs[i] for i, pi_id in enumerate(pairwise_returns)
            }

        return meta_policy
