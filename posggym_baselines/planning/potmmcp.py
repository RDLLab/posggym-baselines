import math
import random
import time
from typing import Dict, Optional, Tuple

import posggym.model as M
from posggym.agents.policy import Policy, PolicyState
from posggym.utils.history import JointHistory

import posggym_baselines.planning.belief as B
from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.mcts import MCTS
from posggym_baselines.planning.node import ObsNode
from posggym_baselines.planning.other_policy import OtherAgentPolicy
from posggym_baselines.planning.search_policy import SearchPolicy, SearchPolicyWrapper


class POTMMCP(MCTS):
    """Partially Observable Type-Based Multi-Agent Monte-Carlo Planning."""

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: MCTSConfig,
        other_agent_policies: Dict[str, OtherAgentPolicy],
        search_policy: "POTMMCPMetaPolicy",
    ):
        super().__init__(model, agent_id, config, other_agent_policies, search_policy)
        assert self.num_agents == 2, "Currently only supports 2 agents"

    #######################################################
    # UPDATE
    #######################################################

    def _initial_update(self, init_obs: M.ObsType):
        action_node = self.root.add_child(None)
        next_search_policy_state = self._update_policy_state(
            action_node.action,
            init_obs,
            self.search_policy,
            self.root.search_policy_state,
        )
        obs_node = ObsNode(
            action_node,
            init_obs,
            t=self.root.t + 1,
            belief=B.ParticleBelief(self._rng),
            action_probs=self.search_policy.get_expected_action_probs(
                None,  # uniform prior over other agent policies
                next_search_policy_state,
            ),
            search_policy_state=next_search_policy_state,
            init_value=0.0,
            init_visits=0,
        )
        action_node.add_child_node(obs_node)

        try:
            # check if model has implemented get_agent_initial_belief
            self.model.sample_agent_initial_state(self.agent_id, init_obs)
            rejection_sample = False
        except NotImplementedError:
            rejection_sample = True

        hps_b0 = B.ParticleBelief(self._rng)
        init_actions = {i: None for i in self.model.possible_agents}
        while hps_b0.size() < self.config.num_particles + self.config.extra_particles:
            # do rejection sampling from initial belief with initial obs
            if rejection_sample:
                state = self.model.sample_initial_state()
                joint_obs = self.model.sample_initial_obs(state)
                if joint_obs[self.agent_id] != init_obs:
                    continue
            else:
                state = self.model.sample_agent_initial_state(self.agent_id, init_obs)
                joint_obs = self.model.sample_initial_obs(state)
                joint_obs[self.agent_id] = init_obs

            joint_history = JointHistory.get_init_history(
                self.model.possible_agents, joint_obs
            )
            other_agent_policy_state = {
                j: self.other_agent_policies[j].sample_initial_state()
                for j in self.model.possible_agents
                if j != self.agent_id
            }
            other_agent_policy_state = self._update_other_agent_policies(
                init_actions, joint_obs, other_agent_policy_state
            )
            hps_b0.add_particle(
                B.HistoryPolicyState(
                    state,
                    joint_history,
                    other_agent_policy_state,
                    t=1,
                )
            )

        obs_node.belief = hps_b0
        self.root = obs_node
        self.root.parent = None

    def _update(self, action: M.ActType, obs: M.ObsType):
        self._log_debug("Pruning histories")
        # Get new root node
        try:
            action_node = self.root.get_child(action)
        except AssertionError as ex:
            if self.root.is_absorbing:
                action_node = self.root.add_child(action)
            else:
                raise ex

        try:
            obs_node = action_node.get_child(obs)
        except AssertionError:
            # Obs node not found
            # Add obs node with uniform policy prior
            # This will be updated in the course of doing simulations
            next_search_policy_state = self._update_policy_state(
                action_node.action,
                obs,
                self.search_policy,
                self.root.search_policy_state,
            )
            obs_node = ObsNode(
                action_node,
                obs,
                t=self.root.t + 1,
                belief=B.ParticleBelief(self._rng),
                action_probs=self.search_policy.get_expected_action_probs(
                    None,  # uniform prior over other agent policies
                    next_search_policy_state,
                ),
                search_policy_state=next_search_policy_state,
                init_value=0.0,
                init_visits=0,
            )
            action_node.add_child_node(obs_node)

            obs_node.is_absorbing = self.root.is_absorbing

        if obs_node.is_absorbing:
            self._log_debug("Absorbing state reached.")
        else:
            self._log_debug(
                f"Belief size before reinvigoration = {obs_node.belief.size()}"
            )
            self._log_debug(f"Parent belief size = {self.root.belief.size()}")
            self._reinvigorate(obs_node, action, obs)
            self._log_debug(
                f"Belief size after reinvigoration = {obs_node.belief.size()}"
            )

        self.root = obs_node
        # remove reference to parent, effectively pruning dead branches
        obs_node.parent = None

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.ActType:
        if self.root.is_absorbing:
            self._log_debug("Agent in absorbing state. Not running search.")
            return self.action_space[0]

        self._log_info(
            f"Searching for search_time_limit={self.config.search_time_limit}"
        )
        start_time = time.time()

        if len(self.root.children) == 0:
            for action in self.action_space:
                self.root.add_child(action)

        max_search_depth = 0
        n_sims = 0
        while time.time() - start_time < self.config.search_time_limit:
            hps = self.root.belief.sample()
            # root sample policy from meta-policy to use for search policy for this
            # simulation
            sim_search_policy = self.search_policy.sample_policy(hps.policy_state)
            _, search_depth = self._simulate(hps, self.root, 0, sim_search_policy)
            self.root.visits += 1
            max_search_depth = max(max_search_depth, search_depth)
            n_sims += 1

        search_time = time.time() - start_time
        self.step_statistics["search_time"] = search_time
        self.step_statistics["search_depth"] = max_search_depth
        self.step_statistics["num_sims"] = n_sims
        self.step_statistics["min_value"] = self._min_max_stats.minimum
        self.step_statistics["max_value"] = self._min_max_stats.maximum
        self._log_info(f"{search_time=:.2f} {max_search_depth=}")
        if self.config.known_bounds is None:
            self._log_info(
                f"{self._min_max_stats.minimum=:.2f} "
                f"{self._min_max_stats.maximum=:.2f}"
            )
        self._log_info(f"Root node policy prior = {self.root.policy_str()}")

        return self._final_action_selection(self.root)

    def _simulate(
        self,
        hps: B.HistoryPolicyState,
        obs_node: ObsNode,
        depth: int,
        search_policy: Policy,
    ) -> Tuple[float, int]:
        if depth > self.config.depth_limit or obs_node.t > self.step_limit:
            return 0, depth

        if len(obs_node.children) == 0:
            # lead node reached
            for action in self.action_space:
                obs_node.add_child(action)

            # use search policy sampled from meta-policy for this simulation
            # for leaf node evaluations
            leaf_node_value = self._evaluate(
                hps,
                depth,
                search_policy,
                obs_node.search_policy_state[search_policy.policy_id],
            )
            return leaf_node_value, depth

        ego_action = self._search_action_selection(obs_node)
        joint_action = self._get_joint_action(hps, ego_action)

        joint_step = self.model.step(hps.state, joint_action)
        joint_obs = joint_step.observations

        ego_obs = joint_obs[self.agent_id]
        ego_return = joint_step.rewards[self.agent_id]
        ego_done = (
            joint_step.terminations[self.agent_id]
            or joint_step.truncations[self.agent_id]
            or joint_step.all_done
        )

        new_joint_history = hps.history.extend(joint_action, joint_obs)
        next_pi_state = self._update_other_agent_policies(
            joint_action, joint_obs, hps.policy_state
        )
        next_hps = B.HistoryPolicyState(
            joint_step.state, new_joint_history, next_pi_state, hps.t + 1
        )

        action_node = obs_node.get_child(ego_action)

        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
            child_obs_node.visits += 1
            # Add search policy distribution to moving average policy of node
            action_probs = search_policy.get_pi(
                obs_node.search_policy_state[search_policy.policy_id]
            )
            for a, a_prob in action_probs.items():
                old_prob = obs_node.action_probs[a]
                obs_node.action_probs[a] += (a_prob - old_prob) / obs_node.visits
        else:
            # add new obs node
            # get next state for all policies in meta-policy, but only use the sampled
            # policy for generating the initial action probs
            next_search_policy_state = self._update_policy_state(
                ego_action,
                ego_obs,
                self.search_policy,
                obs_node.search_policy_state,
            )
            child_obs_node = ObsNode(
                action_node,
                ego_obs,
                t=obs_node.t + 1,
                belief=B.ParticleBelief(self._rng),
                # use the policy sampled from the meta-policy for this simulation
                action_probs=search_policy.get_pi(
                    next_search_policy_state[search_policy.policy_id]
                ),
                search_policy_state=next_search_policy_state,
                init_value=0.0,
                init_visits=1,
            )
            action_node.add_child_node(child_obs_node)
        child_obs_node.is_absorbing = ego_done
        child_obs_node.belief.add_particle(next_hps)

        max_depth = depth
        if not ego_done:
            future_return, max_depth = self._simulate(
                next_hps, child_obs_node, depth + 1, search_policy
            )
            ego_return += self.config.discount * future_return

        action_node.update(ego_return)
        self._min_max_stats.update(action_node.value)
        return ego_return, max_depth


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
        super().__init__(model, agent_id, "POTMMCPMetaPolicy")
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
            "sample_policy to sample a policy and then get the action distribution "
            "from that."
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
        """Load POTMMCPMetaPolicy from posggym agents meta-policy.

        `meta_policy` is a dictionary mapping from other agent policy ID to a
        distribution over ego agent policy IDs (a dictionary mapping ego agent policy
        ID to it's probability).

        """

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
