import logging
import math
import random
import time
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import posggym.model as M
from posggym.agents.policy import Policy, PolicyState
from posggym.utils.history import AgentHistory, JointHistory

import posggym_baselines.planning.belief as B
from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.node import ActionNode, ObsNode
from posggym_baselines.planning.other_policy import OtherAgentPolicy
from posggym_baselines.planning.search_policy import RandomSearchPolicy, SearchPolicy
from posggym_baselines.planning.utils import MinMaxStats, PlanningStatTracker


class INTMCP:
    """Interactive Nested Tree Monte-Carlo Planning (I-NTMCP).

    I-NTMCP models the problem as an I-POMDP and uses MCTS to generate the policy at
    each nesting level. I-NTMCP is equivalent to CI-I-POMCP when the lower level models
    are solved via MCTS, and assuming no explicit communication (communication can still
    occur via actions and observations but it is not explicitly modelled as separate
    actions and observations).

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: MCTSConfig,
        nesting_level: int,
        other_agent_policies: Optional[Dict[str, "INTMCP"]],
        search_policies: Dict[str, SearchPolicy],
    ):
        if other_agent_policies is None:
            other_agent_policies = {}

        self.model = model
        self.agent_id = agent_id
        self.config = config
        self.nesting_level = nesting_level
        self.num_agents = len(model.possible_agents)

        assert nesting_level == 0 or len(other_agent_policies) == self.num_agents - 1
        assert agent_id not in other_agent_policies
        self.other_agent_policies = other_agent_policies

        assert all(i in search_policies for i in model.possible_agents)
        self.search_policies = search_policies

        assert all(
            isinstance(model.action_spaces[i], gym.spaces.Discrete)
            for i in model.possible_agents
        )
        self.action_spaces = {
            i: list(range(model.action_spaces[i].n)) for i in model.possible_agents
        }

        self._min_max_stats = MinMaxStats(config.known_bounds)
        self._rng = random.Random(config.seed)

        if config.step_limit is not None:
            self.step_limit = config.step_limit
        elif model.spec is not None:
            self.step_limit = model.spec.max_episode_steps
        else:
            self.step_limit = None

        self._reinvigorator = B.BeliefRejectionSampler(
            model, config.state_belief_only, sample_limit=4 * self.config.num_particles
        )

        if config.action_selection == "pucb":
            self._search_action_selection = self.pucb_action_selection
            self._final_action_selection = self.max_visit_action_selection
        elif config.action_selection == "ucb":
            self._search_action_selection = self.ucb_action_selection
            self._final_action_selection = self.max_value_action_selection
        else:
            self._search_action_selection = self.min_visit_action_selection
            self._final_action_selection = self.max_value_action_selection

        self.root = ObsNode(
            parent=None,
            obs=None,
            t=0,
            belief=B.ParticleBelief(self._rng),
            action_probs={None: 1.0},
            search_policy_state=self.search_policies[self.agent_id].get_initial_state(),
        )
        self._last_action = None
        self.history = AgentHistory.get_init_history(obs=None)

        self._step_num = 0
        self.step_statistics: Dict[str, float] = {}
        self._reset_step_statistics()
        self.stat_tracker = PlanningStatTracker(self)

        self._logger = logging.getLogger()

    #######################################################
    # Step
    #######################################################

    def step(self, obs: M.ObsType) -> M.ActType:
        root = self.traverse(self.history)
        assert self.step_limit is None or root.t <= self.step_limit
        if root.is_absorbing:
            for k in self.step_statistics:
                self.step_statistics[k] = np.nan
            return self._last_action

        self._reset_step_statistics()

        self._log_info(f"Step {self._step_num} obs={obs}")
        self.update(self._last_action, obs)

        self._last_action = self.get_action()
        self._step_num += 1

        self.step_statistics.update(self._collect_nested_statistics())
        self.stat_tracker.step()

        return self._last_action

    def _collect_nested_statistics(self) -> Dict:
        # This functions adds up statistics of each NST in the hierarchy
        # Only adds statistics that are collected at each level, so doesn't
        # add up 'search_time' or 'update_time' which are only collected
        # in the top tree
        stats = {"reinvigoration_time": self.step_statistics["reinvigoration_time"]}
        if self.nesting_level > 0:
            for pi in self.other_agent_policies.values():
                nested_stats = pi._collect_nested_statistics()
                for key in stats:
                    if key in ("search_depth", "min_value", "max_value"):
                        # ignore search depth from lower levels
                        continue
                    stats[key] += nested_stats[key]
        return stats

    #######################################################
    # RESET
    #######################################################

    def reset(self):
        self._log_info("Reset")
        self.stat_tracker.reset_episode()
        self._step_num = 0
        self._min_max_stats = MinMaxStats(self.config.known_bounds)

        self.root = ObsNode(
            parent=None,
            obs=None,
            t=0,
            belief=B.ParticleBelief(self._rng),
            action_probs={None: 1.0},
            search_policy_state=self.search_policies[self.agent_id].get_initial_state(),
        )
        self._last_action = None
        self.history = AgentHistory.get_init_history(obs=None)

        for pi in self.other_agent_policies.values():
            pi.reset()

        self._reset_step_statistics()

    def _reset_step_statistics(self):
        self.step_statistics = {
            "search_time": 0.0,
            "update_time": 0.0,
            "reinvigoration_time": 0.0,
            "evaluation_time": 0.0,
            "policy_calls": 0,
            "inference_time": 0.0,
            "search_depth": 0,
            "num_sims": 0,
            "min_value": self._min_max_stats.minimum,
            "max_value": self._min_max_stats.maximum,
        }

    #######################################################
    # UPDATE
    #######################################################

    def update(self, action: M.ActType, obs: M.ObsType):
        root = self.traverse(self.history)
        self._log_info(f"Step {root.t} update for a={action} o={obs}")
        if root.is_absorbing:
            return

        start_time = time.time()
        if root.t == 0:
            self.history = AgentHistory.get_init_history(obs)
            self._initial_nested_update({self.history: 1.0})
        else:
            self.history = self.history.extend(action, obs)
            self._nested_update({self.history: 1.0}, len(self.history.history))

        update_time = time.time() - start_time
        self.step_statistics["update_time"] = update_time
        self._log_info(f"Update time = {update_time:.4f}s")

    def _initial_nested_update(self, history_dist: Dict[AgentHistory, float]):
        try:
            # check if model has implemented get_agent_initial_belief
            hist = next(iter(history_dist))
            _, init_obs = hist.get_last_step()
            self.model.sample_agent_initial_state(self.agent_id, init_obs)
            rejection_sample = False
        except NotImplementedError:
            rejection_sample = True

        # policy state is purely function of joint history
        init_actions = {i: None for i in self.model.possible_agents}
        for hist, h_prob in history_dist.items():
            h_node = self.traverse(hist)
            _, init_obs = hist.get_last_step()

            hps_b0 = B.ParticleBelief(self._rng)
            while hps_b0.size() < (
                h_prob * (self.config.num_particles + self.config.extra_particles)
            ):
                if rejection_sample:
                    state = self.model.sample_initial_state()
                    joint_obs = self.model.sample_initial_obs(state)
                    if joint_obs[self.agent_id] != init_obs:
                        continue
                else:
                    state = self.model.sample_agent_initial_state(
                        self.agent_id, init_obs
                    )
                    joint_obs = self.model.sample_initial_obs(state)
                    joint_obs[self.agent_id] = init_obs

                joint_history = JointHistory.get_init_history(
                    self.model.possible_agents, joint_obs
                )

                # store search policy state for each other agent
                # this is used during rollouts and belief reinvigoration
                # TODO remove this if using truncated search?
                policy_state = {
                    j: self.search_policies[j].get_initial_state()
                    for j in self.model.possible_agents
                    if j != self.agent_id
                }
                policy_state = self._update_other_agent_search_policies(
                    init_actions, joint_obs, policy_state
                )

                hps_b0.add_particle(
                    B.HistoryPolicyState(
                        state,
                        joint_history,
                        policy_state,
                        t=1,
                    )
                )
            h_node.belief = hps_b0

        if self.nesting_level > 0:
            nested_histories = self.get_nested_history_dist(history_dist)
            for i, pi in self.other_agent_policies.items():
                pi._initial_nested_update(nested_histories[i])

    def _nested_update(self, history_dist: Dict[AgentHistory, float], current_t: int):
        self._log_debug("Pruning unused nodes from tree")
        # traverse all nodes in tree up to current step
        for action_node in self.root.children:
            for obs_node in action_node.children:
                h = AgentHistory(((action_node.action, obs_node.obs),))
                self._prune_traverse(obs_node, h, history_dist, current_t)

        self._log_debug(f"Reinvigorating beliefs for num_histories={len(history_dist)}")
        target_total_particles = self.config.num_particles + self.config.extra_particles
        for hist, h_prob in history_dist.items():
            h_node = self.traverse(hist)
            if h_node.is_absorbing:
                continue

            action, obs = hist.get_last_step()
            self._reinvigorate(
                h_node,
                action,
                obs,
                target_node_size=math.ceil(h_prob * target_total_particles),
            )

        if self.nesting_level > 0:
            nested_histories = self.get_nested_history_dist(history_dist)
            for i, pi in self.other_agent_policies.items():
                pi._nested_update(nested_histories[i], current_t)

    def _prune_traverse(
        self,
        obs_node: ObsNode,
        node_history: AgentHistory,
        history_dist: Dict[AgentHistory, float],
        current_t: int,
    ):
        """Recursively Traverse and prune histories from tree"""
        if len(node_history.history) == current_t:
            if node_history not in history_dist:
                # pruning history
                del obs_node
            return

        for action_node in obs_node.children:
            for child_obs_node in action_node.children:
                history_tp1 = node_history.extend(
                    action_node.action, child_obs_node.obs
                )

                if len(history_tp1.history) <= current_t - 2:
                    # clear particles from unused nodes to reduce mem usage
                    # nodes more than two steps behind current step are no longer used
                    # except for traversing tree
                    obs_node.clear_belief()

                self._prune_traverse(
                    child_obs_node, history_tp1, history_dist, current_t
                )

    def get_nested_history_dist(
        self,
        histories: Dict[AgentHistory, float],
    ) -> Dict[int, Dict[AgentHistory, float]]:
        """Get distribution over nested histories given higher level distribution."""
        nested_histories = {}
        for hist, h_prob in histories.items():
            h_node = self.traverse(hist)
            belief_size = h_node.belief.size()
            particles = h_node.belief.particles

            for i in self.other_agent_policies:
                if i not in nested_histories:
                    nested_histories[i] = {}
                nested_histories_i = nested_histories[i]

                h_count = {}
                for hps in particles:
                    h = hps.history.get_agent_history(i)
                    h_count[h] = h_count.get(h, 0) + 1

                for h, count in h_count.items():
                    nested_histories_i[h] = nested_histories_i.get(h, 0) + h_prob * (
                        count / belief_size
                    )

        return nested_histories

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.ActType:
        # Note: this function should only be called from top level tree
        # recursive calls during search are done via the _nested_sim function
        root = self.traverse(self.history)
        if root.is_absorbing:
            self._log_debug("Agent in absorbing state. Not running search.")
            return self.action_spaces[self.agent_id][0]

        self._log_info(
            f"Searching for search_time_limit={self.config.search_time_limit}"
        )
        start_time = time.time()

        per_level_search_time_limit = self.config.search_time_limit / (
            self.nesting_level + 1
        )
        n_sims = 0
        for level in range(self.nesting_level + 1):
            self._log_info(f"Searching level {level=}")
            n_sims += self.step_statistics["num_sims"]

            level_start_time = time.time()
            while time.time() - level_start_time < per_level_search_time_limit:
                self._nested_sim(self.history, level, True)
                n_sims += 1

        search_time = time.time() - start_time
        self.step_statistics["search_time"] = search_time
        self.step_statistics["num_sims"] = n_sims
        self.step_statistics["min_value"] = self._min_max_stats.minimum
        self.step_statistics["max_value"] = self._min_max_stats.maximum
        max_search_depth = self.step_statistics["search_depth"]
        self._log_info(f"{search_time=:.2f} {max_search_depth=} {n_sims=}")
        if self.config.known_bounds is None:
            self._log_info(
                f"{self._min_max_stats.minimum=:.2f} "
                f"{self._min_max_stats.maximum=:.2f}"
            )
        self._log_info(f"Root node policy prior = {root.policy_str()}")

        return self._final_action_selection(root)

    def _nested_sim(
        self, history: AgentHistory, search_level: int, top_level: bool = False
    ):
        """Run nested simulation on this policy tree starting from history"""
        self._log_debug(f"nested_sim: {search_level=}")

        root = self.traverse(history)
        if len(root.children) == 0:
            for action in self.action_spaces[self.agent_id]:
                root.add_child(action)

        if root.belief.size() == 0 or (
            top_level and root.belief.size() < self.config.extra_particles
        ):
            # handle depleted node
            self._log_debug(f"depleted {root.belief.size()=} {top_level=}")
            self._reinvigorate(
                root,
                root.parent.action,
                root.obs,
                target_node_size=self.config.extra_particles,
            )

        hps = root.belief.sample()
        if self.nesting_level > search_level:
            for i, pi in self.other_agent_policies.items():
                pi._nested_sim(hps.history.get_agent_history(i), search_level, False)
        else:
            _, search_depth = self._simulate(hps, root, 0)
            root.visits += 1
            self.step_statistics["search_depth"] = max(
                self.step_statistics["search_depth"], search_depth
            )

    def _simulate(
        self,
        hps: B.HistoryPolicyState,
        obs_node: ObsNode,
        depth: int,
    ) -> Tuple[float, int]:
        if depth > self.config.depth_limit or (
            self.step_limit is not None and obs_node.t + depth > self.step_limit
        ):
            return 0, depth

        if len(obs_node.children) < len(self.action_spaces[self.agent_id]):
            # lead node reached
            # some child action nodes may have been added by parent tree querying
            # so need to check if all actions have been added, and add if not
            for action in self.action_spaces[self.agent_id]:
                if not obs_node.has_child(action):
                    obs_node.add_child(action)
            leaf_node_value = self._evaluate(hps, depth, obs_node)
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
        next_pi_state = self._update_other_agent_search_policies(
            joint_action, joint_obs, hps.policy_state
        )
        next_hps = B.HistoryPolicyState(
            joint_step.state, new_joint_history, next_pi_state, hps.t + 1
        )

        action_node = obs_node.get_child(ego_action)

        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
            child_obs_node.visits += 1

            if self.config.action_selection == "pucb":
                # Add search policy distribution to moving average policy of node
                action_probs = self.search_policies[self.agent_id].get_pi(
                    child_obs_node.search_policy_state
                )
                for a, a_prob in action_probs.items():
                    old_prob = child_obs_node.action_probs[a]
                    child_obs_node.action_probs[a] += (
                        a_prob - old_prob
                    ) / child_obs_node.visits
        else:
            child_obs_node = self._add_obs_node(action_node, ego_obs, init_visits=1)
        child_obs_node.is_absorbing = ego_done
        child_obs_node.belief.add_particle(next_hps)

        max_depth = depth
        if not ego_done:
            future_return, max_depth = self._simulate(
                next_hps, child_obs_node, depth + 1
            )
            ego_return += self.config.discount * future_return

        action_node.update(ego_return)
        self._min_max_stats.update(action_node.value)
        return ego_return, max_depth

    def _evaluate(
        self,
        hps: B.HistoryPolicyState,
        depth: int,
        obs_node: ObsNode,
    ) -> float:
        start_time = time.time()
        if self.config.truncated:
            try:
                v = self.search_policies[self.agent_id].get_value(
                    obs_node.search_policy_state
                )
            except NotImplementedError as e:
                if self.config.use_rollout_if_no_value:
                    search_policy_states = dict(hps.policy_state)
                    search_policy_states[self.agent_id] = obs_node.search_policy_state
                    v = self._rollout(
                        hps, depth, self.search_policies, search_policy_states
                    )
                else:
                    raise e
        else:
            search_policy_states = dict(hps.policy_state)
            search_policy_states[self.agent_id] = obs_node.search_policy_state
            v = self._rollout(hps, depth, self.search_policies, search_policy_states)
        self.step_statistics["evaluation_time"] += time.time() - start_time
        return v

    def _rollout(
        self,
        hps: B.HistoryPolicyState,
        depth: int,
        rollout_policies: Dict[str, Policy],
        rollout_policies_states: Dict[str, PolicyState],
    ) -> float:
        if depth > self.config.depth_limit or (
            self.step_limit is not None and hps.t > self.step_limit
        ):
            return 0

        joint_action = {
            i: rollout_policies[i].sample_action(rollout_policies_states[i])
            for i in self.model.possible_agents
        }

        joint_step = self.model.step(hps.state, joint_action)
        joint_obs = joint_step.observations
        reward = joint_step.rewards[self.agent_id]

        if (
            joint_step.terminations[self.agent_id]
            or joint_step.truncations[self.agent_id]
            or joint_step.all_done
        ):
            return reward

        next_pi_state = self._update_other_agent_search_policies(
            joint_action, joint_obs, hps.policy_state
        )
        # history is None as it is not used during rollouts
        next_hps = B.HistoryPolicyState(
            joint_step.state, None, next_pi_state, hps.t + 1
        )

        next_rollout_policies_states = {
            i: self._update_policy_state(
                joint_action[i],
                joint_obs[i],
                rollout_policies[i],
                rollout_policies_states[i],
            )
            for i in self.model.possible_agents
        }

        future_return = self._rollout(
            next_hps, depth + 1, rollout_policies, next_rollout_policies_states
        )
        return reward + self.config.discount * future_return

    def _update_policy_state(
        self,
        action: M.ActType,
        obs: M.ObsType,
        policy: Union[Policy, OtherAgentPolicy],
        policy_state: PolicyState,
    ) -> PolicyState:
        # this is just a wrapper around policy.get_next_state but also keeps track of
        # inference time and number of policy calls
        start_time = time.time()
        next_hidden_state = policy.get_next_state(action, obs, policy_state)
        self.step_statistics["inference_time"] += time.time() - start_time
        self.step_statistics["policy_calls"] += 1
        return next_hidden_state

    def _update_other_agent_search_policies(
        self,
        joint_action: Dict[str, Optional[M.ActType]],
        joint_obs: Dict[str, M.ObsType],
        pi_state: Dict[str, PolicyState],
    ) -> Dict[str, PolicyState]:
        next_policy_state = {}
        for i in self.model.possible_agents:
            if i == self.agent_id or i not in joint_action:
                h_t = None
            else:
                h_t = self._update_policy_state(
                    joint_action[i],
                    joint_obs[i],
                    self.search_policies[i],
                    pi_state[i],
                )
            next_policy_state[i] = h_t
        return next_policy_state

    #######################################################
    # ACTION SELECTION
    #######################################################

    def pucb_action_selection(self, obs_node: ObsNode) -> M.ActType:
        """Select action from node using PUCB."""
        if obs_node.visits == 0:
            # sample action using prior policy
            return random.choices(
                list(obs_node.action_probs.keys()),
                weights=list(obs_node.action_probs.values()),
                k=1,
            )[0]

        # add exploration noise to prior
        prior, noise = {}, 1 / len(self.action_spaces[self.agent_id])
        for a in self.action_space[self.agent_id]:
            prior[a] = (
                obs_node.action_probs[a] * (1 - self.config.root_exploration_fraction)
                + noise
            )

        sqrt_n = math.sqrt(obs_node.visits)
        max_v = -float("inf")
        max_action = obs_node.children[0].action
        for action_node in obs_node.children:
            explore_v = (
                self.config.c
                * prior[action_node.action]
                * (sqrt_n / (1 + action_node.visits))
            )
            if action_node.visits > 0:
                action_v = self._min_max_stats.normalize(action_node.value)
            else:
                action_v = 0
            action_v = action_v + explore_v
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.action
        return max_action

    def ucb_action_selection(self, obs_node: ObsNode) -> M.ActType:
        """Select action from node using UCB."""
        if obs_node.visits == 0:
            return random.choice(self.action_spaces[self.agent_id])

        log_n = math.log(obs_node.visits)

        max_v = -float("inf")
        max_action = obs_node.children[0].action
        for action_node in obs_node.children:
            if action_node.visits == 0:
                return action_node.action
            explore_v = self.config.c * math.sqrt(log_n / action_node.visits)
            action_v = self._min_max_stats.normalize(action_node.value) + explore_v
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.action
        return max_action

    def min_visit_action_selection(self, obs_node: ObsNode) -> M.ActType:
        """Select action from node with least visits.

        Note this guarantees all actions are visited equally +/- 1 when used
        during search.
        """
        if obs_node.visits == 0:
            return random.choice(self.action_spaces[self.agent_id])

        min_n = obs_node.visits + 1
        next_action = obs_node.children[0].action
        for action_node in obs_node.children:
            if action_node.visits < min_n:
                min_n = action_node.visits
                next_action = action_node.action
        return next_action

    def max_visit_action_selection(self, obs_node: ObsNode) -> M.ActType:
        """Select action from node with most visits.

        Breaks ties randomly.
        """
        if obs_node.visits == 0:
            return random.choice(self.action_spaces[self.agent_id])

        max_actions = []
        max_visits = 0
        for a_node in obs_node.children:
            if a_node.visits == max_visits:
                max_actions.append(a_node.action)
            elif a_node.visits > max_visits:
                max_visits = a_node.visits
                max_actions = [a_node.action]
        return random.choice(max_actions)

    def max_value_action_selection(self, obs_node: ObsNode) -> M.ActType:
        """Select action from node with maximum value.

        Breaks ties randomly.
        """
        if len(obs_node.children) == 0:
            # Node not expanded so select random action
            return random.choice(self.action_spaces[self.agent_id])

        max_actions = []
        max_value = -float("inf")
        for a_node in obs_node.children:
            if a_node.value == max_value:
                max_actions.append(a_node.action)
            elif a_node.value > max_value:
                max_value = a_node.value
                max_actions = [a_node.action]
        return random.choice(max_actions)

    def _get_joint_action(
        self, hps: B.HistoryPolicyState, ego_action: M.ActType
    ) -> Dict[str, M.ActType]:
        agent_actions = {}
        for i in self.model.possible_agents:
            if i == self.agent_id:
                a_i = ego_action
            elif self.nesting_level == 0 or self.config.state_belief_only:
                # assume other agents policies are stateless, i.e. Random
                a_i = self._rng.choice(self.action_spaces[i])
            else:
                a_i = self.other_agent_policies[i].sample_action(
                    hps.history.get_agent_history(i),
                    self.search_policies[i],
                    hps.policy_state[i],
                )
            agent_actions[i] = a_i
        return agent_actions

    def sample_action(
        self,
        history: AgentHistory,
        search_policy: SearchPolicy,
        search_policy_state: PolicyState,
    ) -> M.ActType:
        """Sample an action from policy given history

        Samples using softmax function (with temperature=1.0) scaled based on sqrt of
        total number of visits of obs_node.

        Uses `search_policy` to sample action if there is no node in the tree
        corresponding to the given history.

        """
        obs_node = self.traverse(history)
        if obs_node.visits == 0 or len(obs_node.children) == 0:
            return search_policy.sample_action(search_policy_state)

        obs_n_sqrt = math.sqrt(obs_node.visits)
        a_probs = [math.exp(a_node.visits / obs_n_sqrt) for a_node in obs_node.children]
        a_probs_sum = sum(a_probs)
        a_probs = [p / a_probs_sum for p in a_probs]

        action_node = random.choices(obs_node.children, weights=a_probs)[0]
        return action_node.action

    #######################################################
    # GENERAL METHODS
    #######################################################

    def traverse(self, history: AgentHistory) -> ObsNode:
        """Traverse policy tree and return node corresponding to history."""
        obs_node = self.root
        for a, o in history:
            try:
                action_node = obs_node.get_child(a)
            except AssertionError:
                action_node = obs_node.add_child(a)
            try:
                obs_node = action_node.get_child(o)
            except AssertionError:
                obs_node = self._add_obs_node(action_node, o, init_visits=0)
        return obs_node

    def _add_obs_node(
        self,
        parent: ActionNode,
        obs: M.ObsType,
        init_visits: int = 0,
    ) -> ObsNode:
        search_policy = self.search_policies[self.agent_id]
        next_search_policy_state = self._update_policy_state(
            parent.action,
            obs,
            search_policy,
            parent.parent.search_policy_state,
        )

        obs_node = ObsNode(
            parent,
            obs,
            t=parent.t + 1,
            belief=B.ParticleBelief(self._rng),
            action_probs=search_policy.get_pi(next_search_policy_state),
            search_policy_state=next_search_policy_state,
            init_value=0.0,
            init_visits=init_visits,
        )
        parent.children.append(obs_node)

        return obs_node

    #######################################################
    # BELIEF REINVIGORATION
    #######################################################

    def _reinvigorate(
        self,
        obs_node: ObsNode,
        action: M.ActType,
        obs: M.ObsType,
        target_node_size: Optional[int] = None,
    ):
        """Reinvigoration belief associated to given history.

        The general reinvigoration process:
        1. check belief needs to be reinvigorated (e.g. it's not a root belief)
        2. Reinvigorate node using rejection sampling/custom function for fixed
           number of tries
        3. if desired number of particles not sampled using rejection sampling/
           custom function then sample remaining particles using sampling
           without rejection
        """
        start_time = time.time()

        belief_size = obs_node.belief.size()
        if belief_size is None:
            # root belief
            return

        if target_node_size is None:
            particles_to_add = self.config.num_particles + self.config.extra_particles
        else:
            particles_to_add = target_node_size
        particles_to_add -= belief_size

        if particles_to_add <= 0:
            return

        parent_obs_node = obs_node.parent.parent
        assert parent_obs_node is not None

        self._reinvigorator.reinvigorate(
            self.agent_id,
            obs_node.belief,
            action,
            obs,
            num_particles=particles_to_add,
            parent_belief=parent_obs_node.belief,
            joint_action_fn=self._reinvigorate_action_fn,
            joint_update_fn=self._reinvigorate_update_fn,
            **{"use_rejected_samples": True},  # used for rejection sampling
        )

        reinvig_time = time.time() - start_time
        self.step_statistics["reinvigoration_time"] += reinvig_time

    def _reinvigorate_action_fn(
        self, hps: B.HistoryPolicyState, ego_action: M.ActType
    ) -> Dict[str, M.ActType]:
        # sample actions using search policy for each agent, rather than using
        # nested tree
        joint_action = {}
        for i in self.model.possible_agents:
            if i == self.agent_id:
                joint_action[i] = ego_action
            elif self.nesting_level == 0 or self.config.state_belief_only:
                # assume other agents policies are stateless, i.e. Random
                joint_action[i] = self._rng.choice(self.action_spaces[i])
            else:
                joint_action[i] = self.other_agent_policies[i].sample_action(
                    hps.history.get_agent_history(i),
                    self.search_policies[i],
                    hps.policy_state[i],
                )
        return joint_action

    def _reinvigorate_update_fn(
        self,
        hps: B.HistoryPolicyState,
        joint_action: Dict[str, M.ActType],
        joint_obs: Dict[str, M.ObsType],
    ) -> Dict[str, PolicyState]:
        return self._update_other_agent_search_policies(
            joint_action, joint_obs, hps.policy_state
        )

    #######################################################
    # Logging and General methods
    #######################################################

    def close(self):
        """Do any clean-up."""
        for policy in self.search_policies.values():
            policy.close()
        for policy in self.other_agent_policies.values():
            policy.close()

    def _log_info(self, msg: str):
        """Log an info message."""
        self._logger.info(self._format_msg(msg))

    def _log_debug(self, msg: str):
        """Log a debug message."""
        self._logger.debug(self._format_msg(msg))

    def _format_msg(self, msg: str):
        return f"i={self.agent_id} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    @classmethod
    def initialize(
        cls,
        model: M.POSGModel,
        ego_agent_id: str,
        config: MCTSConfig,
        nesting_level: int,
        search_policies: Optional[Dict[int, Dict[str, SearchPolicy]]],
    ) -> "INTMCP":
        """Initialize a new I-NTMCP datastructure

        Includes recursively initializing all sub trees.

        If `search_policies` is None then a RandomSearchPolicy is used for each agent.

        """
        if search_policies is None:
            search_policies = {}
            for level in range(nesting_level + 1):
                search_policies[level] = {}
                for i in model.possible_agents:
                    search_policies[level][i] = RandomSearchPolicy(model, i)

        other_agent_policies = {}
        if nesting_level > 0:
            for i in model.possible_agents:
                if i == ego_agent_id:
                    continue

                other_agent_policies[i] = INTMCP.initialize(
                    model=model,
                    ego_agent_id=i,
                    config=config,
                    nesting_level=nesting_level - 1,
                    search_policies=search_policies,
                )

        intmcp = INTMCP(
            model=model,
            agent_id=ego_agent_id,
            config=config,
            nesting_level=nesting_level,
            other_agent_policies=other_agent_policies,
            search_policies=search_policies[nesting_level],
        )
        return intmcp
