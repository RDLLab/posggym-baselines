import time
from typing import Dict, Tuple

import posggym.model as M
from posggym.utils.history import JointHistory
from posggym.agents.policy import Policy


import posggym_baselines.planning.mcts.belief as B
from posggym_baselines.planning.mcts.core import POMMCP, POMMCPConfig
from posggym_baselines.planning.mcts.node import ObsNode
from posggym_baselines.planning.potmmcp.other_policy import POTMMCPOtherAgentPolicy
from posggym_baselines.planning.potmmcp.search_policy import POTMMCPMetaPolicy


class POTMMCP(POMMCP):
    """Partially Observable Type-Based Multi-Agent Monte-Carlo Planning."""

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: POMMCPConfig,
        other_agent_policies: Dict[str, POTMMCPOtherAgentPolicy],
        search_policy: POTMMCPMetaPolicy,
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
        action_node.children.append(obs_node)

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
                self.num_agents, tuple(joint_obs[i] for i in self.model.possible_agents)
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
            action_node.children.append(obs_node)

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
        self._statistics["search_time"] = search_time
        self._statistics["search_depth"] = max_search_depth
        self._statistics["num_sims"] = n_sims
        self._statistics["min_value"] = self._min_max_stats.minimum
        self._statistics["max_value"] = self._min_max_stats.maximum
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
        if depth > self.config.depth_limit or (
            self.step_limit is not None and obs_node.t + depth > self.step_limit
        ):
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

        new_history = hps.history.extend(
            tuple(joint_action[i] for i in self.model.possible_agents),
            tuple(joint_obs[i] for i in self.model.possible_agents),
        )
        next_pi_state = self._update_other_agent_policies(
            joint_action, joint_obs, hps.policy_state
        )
        next_hps = B.HistoryPolicyState(joint_step.state, new_history, next_pi_state)

        action_node = obs_node.get_child(ego_action)

        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
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
                init_visits=0,
            )
            action_node.children.append(obs_node)
        child_obs_node.visits += 1

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
