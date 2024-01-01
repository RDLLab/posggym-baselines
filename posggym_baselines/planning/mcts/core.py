import math
import random
import time
import logging
from typing import Dict, Optional, Tuple, Union
from collections import namedtuple
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import posggym.model as M
from posggym.utils.history import JointHistory
from posggym.agents.policy import PolicyState, Policy


import posggym_baselines.planning.mcts.belief as B
from posggym_baselines.planning.mcts.node import ActionNode, ObsNode
from posggym_baselines.planning.mcts.other_policy import OtherAgentPolicy
from posggym_baselines.planning.mcts.search_policy import SearchPolicy


KnownBounds = namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats:
    """A class that holds the min-max values of the tree.

    Ref: MuZero pseudocode
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        if known_bounds:
            self.maximum = known_bounds.max
            self.minimum = known_bounds.min
        else:
            self.maximum = -float("inf")
            self.minimum = float("inf")

    def update(self, value: float):
        """Update min and mad values."""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """Normalize value given known min and max values."""
        if self.maximum > self.minimum:
            # Normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def __str__(self):
        return f"MinMaxState: (minimum: {self.minimum}, maximum: {self.maximum})"


@dataclass
class POMMCPConfig:
    """Configuration for POMMCP."""

    discount: float
    search_time_limit: float
    c: float
    truncated: bool
    action_selection: str = "pucb"
    root_exploration_fraction: float = 0.25
    known_bounds: Optional[KnownBounds] = None
    extra_particles_prop: float = 1.0 / 16
    step_limit: Optional[int] = None
    epsilon: float = 0.01
    seed: Optional[int] = None

    num_particles: int = field(init=False)
    extra_particles: int = field(init=False)
    depth_limit: int = field(init=False)

    def __post_init__(self):
        assert self.discount >= 0.0 and self.discount <= 1.0
        assert self.search_time_limit > 0.0
        assert self.c > 0.0
        assert (
            self.root_exploration_fraction >= 0.0
            and self.root_exploration_fraction <= 1.0
        )
        assert self.extra_particles_prop >= 0.0 and self.extra_particles_prop <= 1.0
        assert self.epsilon > 0.0 and self.epsilon < 1.0

        self.action_selection = self.action_selection.lower()
        assert self.action_selection in ["pucb", "ucb", "uniform"]

        self.num_particles = math.ceil(100 * self.search_time_limit)
        self.extra_particles = math.ceil(self.num_particles * self.extra_particles_prop)

        if self.discount == 0.0:
            self.depth_limit = 0
        else:
            self.depth_limit = math.ceil(
                math.log(self.epsilon) / math.log(self.discount)
            )


class POMMCP:
    """Partially Observable Multi-Agent Monte-Carlo Planning.

    The is the base class for the various MCTS based algorithms.

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: POMMCPConfig,
        other_agent_policies: Dict[str, OtherAgentPolicy],
        search_policy: SearchPolicy,
    ):
        self.model = model
        self.agent_id = agent_id
        self.config = config
        self.num_agents = len(model.possible_agents)
        assert len(other_agent_policies) == self.num_agents - 1
        assert agent_id not in other_agent_policies
        self.other_agent_policies = other_agent_policies
        self.search_policy = search_policy

        assert isinstance(model.action_spaces[agent_id], gym.spaces.Discrete)
        num_actions = model.action_spaces[agent_id].n
        self.action_space = list(range(num_actions))

        self._min_max_stats = MinMaxStats(config.known_bounds)
        self._rng = random.Random(config.seed)

        if config.step_limit is not None:
            self.step_limit = config.step_limit
        elif model.spec is not None:
            self.step_limit = model.spec.max_episode_steps
        else:
            self.step_limit = None

        self._reinvigorator = B.BeliefRejectionSampler(
            model, sample_limit=4 * self.config.num_particles
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
            search_policy_state=self.search_policy.get_initial_state(),
        )
        self._last_action = None

        self._step_num = 0
        self._statistics: Dict[str, float] = {}
        self._reset_step_statistics()

        self._logger = logging.getLogger()

    #######################################################
    # Step
    #######################################################

    def step(self, obs: M.ObsType) -> M.ActType:
        assert self.step_limit is None or self.root.t <= self.step_limit
        if self.root.is_absorbing:
            for k in self._statistics:
                self._statistics[k] = np.nan
            return self._last_action

        self._reset_step_statistics()

        self._log_info(f"Step {self._step_num} obs={obs}")
        self.update(self._last_action, obs)

        self._last_action = self.get_action()
        self._step_num += 1

        return self._last_action

    #######################################################
    # RESET
    #######################################################

    def reset(self):
        self._log_info("Reset")
        self._step_num = 0
        self._min_max_stats = MinMaxStats(self.config.known_bounds)
        self._reset_step_statistics()

        self.root = ObsNode(
            parent=None,
            obs=None,
            t=0,
            belief=B.ParticleBelief(self._rng),
            action_probs={None: 1.0},
            search_policy_state=self.search_policy.get_initial_state(),
        )
        self._last_action = None

    def _reset_step_statistics(self):
        self._statistics = {
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
        self._log_info(f"Step {self.root.t} update for a={action} o={obs}")
        if self.root.is_absorbing:
            return

        start_time = time.time()
        if self.root.t == 0:
            self._last_action = None
            self._initial_update(obs)
        else:
            self._update(action, obs)

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time
        self._log_info(f"Update time = {update_time:.4f}s")

    def _initial_update(self, init_obs: M.ObsType):
        action_node = self.root.add_child(None)
        obs_node = self._add_obs_node(action_node, init_obs, init_visits=0)

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
            policy_state = {
                j: self.other_agent_policies[j].sample_initial_state()
                for j in self.model.possible_agents
                if j != self.agent_id
            }
            policy_state = self._update_other_agent_policies(
                init_actions, joint_obs, policy_state
            )
            hps_b0.add_particle(
                B.HistoryPolicyState(
                    state,
                    joint_history,
                    policy_state,
                )
            )

        obs_node.belief = hps_b0
        self.root = obs_node
        self.root.parent = None

    def _update(self, action: M.ActType, obs: M.ObsType):
        self._log_debug("Pruning histories")
        # Get new root node
        try:
            a_node = self.root.get_child(action)
        except AssertionError as ex:
            if self.root.is_absorbing:
                a_node = self.root.add_child(action)
            else:
                raise ex

        try:
            obs_node = a_node.get_child(obs)
        except AssertionError:
            # Obs node not found
            # Add obs node with uniform policy prior
            # This will be updated in the course of doing simulations
            obs_node = self._add_obs_node(a_node, obs, init_visits=0)
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
            _, search_depth = self._simulate(hps, self.root, 0, self.search_policy)
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
            leaf_node_value = self._evaluate(
                hps,
                depth,
                search_policy,
                obs_node.search_policy_state,
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
            self._update_obs_node(child_obs_node, search_policy)
        else:
            child_obs_node = self._add_obs_node(action_node, ego_obs, init_visits=1)
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

    def _evaluate(
        self,
        hps: B.HistoryPolicyState,
        depth: int,
        rollout_policy: Policy,
        rollout_policy_state: PolicyState,
    ) -> float:
        start_time = time.time()
        if self.config.truncated:
            v = rollout_policy.get_value(rollout_policy_state)
        else:
            v = self._rollout(hps, depth, rollout_policy, rollout_policy_state)
        self._statistics["evaluation_time"] += time.time() - start_time
        return v

    def _rollout(
        self,
        hps: B.HistoryPolicyState,
        depth: int,
        rollout_policy: Policy,
        rollout_policy_state: PolicyState,
    ) -> float:
        if depth > self.config.depth_limit or (
            self.step_limit is not None and hps.t > self.step_limit
        ):
            return 0

        ego_action = rollout_policy.sample_action(rollout_policy_state)
        joint_action = self._get_joint_action(hps, ego_action)

        joint_step = self.model.step(hps.state, joint_action)
        joint_obs = joint_step.observations
        reward = joint_step.rewards[self.agent_id]

        if (
            joint_step.terminations[self.agent_id]
            or joint_step.truncations[self.agent_id]
            or joint_step.all_done
        ):
            return reward

        new_history = hps.history.extend(
            tuple(joint_action[i] for i in self.model.possible_agents),
            tuple(joint_obs[i] for i in self.model.possible_agents),
        )
        next_pi_state = self._update_other_agent_policies(
            joint_action, joint_obs, hps.policy_state
        )
        next_hps = B.HistoryPolicyState(joint_step.state, new_history, next_pi_state)

        next_rollout_policy_state = self._update_policy_state(
            joint_action[self.agent_id],
            joint_obs[self.agent_id],
            rollout_policy,
            rollout_policy_state,
        )

        future_return = self._rollout(
            next_hps, depth + 1, rollout_policy, next_rollout_policy_state
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
        self._statistics["inference_time"] += time.time() - start_time
        self._statistics["policy_calls"] += 1
        return next_hidden_state

    def _update_other_agent_policies(
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
                    self.other_agent_policies[i],
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
        prior, noise = {}, 1 / len(self.action_space)
        for a in self.action_space:
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
            return random.choice(self.action_space)

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
            return random.choice(self.action_space)

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
            return random.choice(self.action_space)

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
            return random.choice(self.action_space)

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
            else:
                a_i = self.other_agent_policies[i].sample_action(hps.policy_state[i])
            agent_actions[i] = a_i
        return agent_actions

    def get_pi(self) -> Dict[M.ActType, float]:
        """Get agent's distribution over actions at root of tree.

        Returns the softmax distribution over actions with temperature=1.0.
        This is used as it incorporates uncertainty based on visit counts for
        a given history.
        """
        if self.root.n == 0 or len(self.root.children) == 0:
            # uniform policy
            num_actions = len(self.action_space)
            pi = {a: 1.0 / num_actions for a in self.action_space}
            return pi

        obs_n_sqrt = math.sqrt(self.root.n)
        temp = 1.0
        pi = {
            a_node.h: math.exp(a_node.n**temp / obs_n_sqrt)
            for a_node in self.root.children
        }

        a_probs_sum = sum(pi.values())
        for a in self.action_space:
            if a not in pi:
                pi[a] = 0.0
            pi[a] /= a_probs_sum

        return pi

    #######################################################
    # GENERAL METHODS
    #######################################################

    def _add_obs_node(
        self,
        parent: ActionNode,
        obs: M.ObsType,
        init_visits: int = 0,
    ) -> ObsNode:
        next_search_policy_state = self._update_policy_state(
            parent.action,
            obs,
            self.search_policy,
            parent.parent.search_policy_state,
        )

        obs_node = ObsNode(
            parent,
            obs,
            t=parent.t + 1,
            belief=B.ParticleBelief(self._rng),
            action_probs=self.search_policy.get_pi(next_search_policy_state),
            search_policy_state=next_search_policy_state,
            init_value=0.0,
            init_visits=init_visits,
        )
        parent.children.append(obs_node)

        return obs_node

    def _update_obs_node(self, obs_node: ObsNode, search_policy: Policy):
        obs_node.visits += 1
        # Add search policy distribution to moving average policy of node
        action_probs = search_policy.get_pi(obs_node.search_policy_state)
        for a, a_prob in action_probs.items():
            old_prob = obs_node.action_probs[a]
            obs_node.action_probs[a] += (a_prob - old_prob) / obs_node.visits

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
        self._statistics["reinvigoration_time"] += reinvig_time

    def _reinvigorate_action_fn(
        self, hps: B.HistoryPolicyState, ego_action: M.ActType
    ) -> Dict[str, M.ActType]:
        return self._get_joint_action(hps, ego_action)

    def _reinvigorate_update_fn(
        self,
        hps: B.HistoryPolicyState,
        joint_action: Dict[str, M.ActType],
        joint_obs: Dict[str, M.ObsType],
    ) -> Dict[str, PolicyState]:
        return self._update_other_agent_policies(
            joint_action, joint_obs, hps.policy_state
        )

    #######################################################
    # Logging and General methods
    #######################################################

    def close(self):
        """Do any clean-up."""
        self.search_policy.close()
        for policy in self.other_agent_policies.values():
            policy.close()

    def _log_info(self, msg: str):
        """Log an info message."""
        self._logger.log(logging.INFO - 1, self._format_msg(msg))

    def _log_debug(self, msg: str):
        """Log a debug message."""
        self._logger.debug(self._format_msg(msg))

    def _format_msg(self, msg: str):
        return f"i={self.agent_id} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__}"
