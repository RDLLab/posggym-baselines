from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import posggym.agents as pga
import posggym.model as M
from posggym.agents.wrappers import AgentEnvWrapper
from posggym.utils.history import AgentHistory

from posggym_baselines.planning.other_policy import OtherAgentPolicy

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


class PlanningStatTracker:
    """Tracks MCTS planning statistics."""

    # The list of keys to track from the policies.statistics property
    STAT_KEYS = [
        "search_time",
        "update_time",
        "reinvigoration_time",
        "evaluation_time",
        "policy_calls",
        "inference_time",
        "search_depth",
        "num_sims",
        "mem_usage",
        "min_value",
        "max_value",
    ]

    # The list of keys to track max values for, otherwise we track mean values
    MAX_STATS = {"mem_usage"}

    def __init__(self, planner, track_overall: bool = True):
        self.planner = planner
        self.track_overall = track_overall

        self._current_steps = 0
        self._current_stats: Dict[str, List[float]] = {k: [] for k in self.STAT_KEYS}

        self._num_episodes = 0
        self._all_steps: List[int] = []
        self._all_stats: Dict[str, List[float]] = {k: [] for k in self.STAT_KEYS}

        self.reset()

    def step(self):
        self._current_steps += 1
        for k in self.STAT_KEYS:
            self._current_stats[k].append(self.planner.step_statistics.get(k, np.nan))

    def reset(self):
        """Reset tracker."""
        self.reset_episode()
        self._num_episodes = 0
        self._all_steps = []
        self._all_stats = {k: [] for k in self.STAT_KEYS}

    def reset_episode(self):
        """Reset for new episode.

        Takes care of summarizing results from any previous episode.
        This should be called at the beginning of each episode, or after finishing
        final episode.
        """
        if self._current_steps == 0:
            return

        self._num_episodes += 1
        if self.track_overall:
            self._all_steps.append(self._current_steps)
            for k in self.STAT_KEYS:
                if k in self.MAX_STATS:
                    self._all_stats[k].append(np.nanmax(self._current_stats[k]))
                else:
                    self._all_stats[k].append(np.nanmean(self._current_stats[k]))

        self._current_steps = 0
        self._current_stats = {k: [] for k in self.STAT_KEYS}

    def get_episode(self) -> Dict[str, float]:
        """Get statistics for current episode."""
        stats = {}
        for key, step_times in self._current_stats.items():
            if len(step_times) == 0 or np.isnan(np.sum(step_times)):
                val = np.nan
            elif key in self.MAX_STATS:
                val = np.nanmax(step_times, axis=0)
            else:
                val = np.nanmean(step_times, axis=0)
            stats[key] = val
        return stats

    def get(self) -> Dict[str, float]:
        """Get statistics for all episodes."""
        stats = {}
        for key, values in self._all_stats.items():
            if len(values) == 0 or np.isnan(np.sum(values)):
                mean_val = np.nan
                std_val = np.nan
            elif key in self.MAX_STATS:
                # get max and std values across episodes
                mean_val = np.nanmax(values, axis=0)
                std_val = np.nanstd(values, axis=0)
            else:
                mean_val = np.nanmean(values, axis=0)
                std_val = np.nanstd(values, axis=0)
            stats[f"{key}_mean"] = mean_val
            stats[f"{key}_std"] = std_val
        return stats


class BeliefStatTracker:
    """Tracks statistics about the beliefs of a planner.

    Can be used to track:

    - `policy`: belief accuracy of other agents' policy ID/type
    - `action`: belief accuracy of other agents' action
    - `state`: belief accuracy of state of the environment
    - `history`: belief accuracy of other agents' history

    Note, this class should be reset at the beginning of each episode using either
    `reset_episode` (if `track_overall=True`) or `reset`. Importantly, the
    BeliefStatTracker should be reset AFTER the environment has been reset, this
    ensures the initial observations for the other agent's have been generated by the
    environment.

    The `step` method should be called after `planner.step()` and `env.step()`
    functions have called. This ensures that the planner's belief has been updated, and
    similarly the environment's `last_actions`, `last_obs`, and state have been updated.

    """

    TRACKABLE_BELIEF_STATS = ["policy", "action", "state", "history"]

    def __init__(
        self,
        planner,
        env: AgentEnvWrapper,
        track_overall: bool = True,
        stats_to_track: List[str] = ["policy", "action", "state", "history"],
    ):
        assert all(k in self.TRACKABLE_BELIEF_STATS for k in stats_to_track)
        if any(k in stats_to_track for k in ["policy", "action"]):
            assert all(
                isinstance(pi, OtherAgentPolicy)
                for pi in planner.other_agent_policies.values()
            )

        self.planner = planner
        self.env = env
        self.track_overall = track_overall
        self.stats_to_track = stats_to_track
        self.other_agent_ids = [
            i for i in planner.model.possible_agents if i != planner.agent_id
        ]

        self._current_steps = 0
        self._current_state = None
        self._current_other_agent_histories = {
            i: AgentHistory.get_init_history() for i in self.other_agent_ids
        }
        self._current_stats: Dict[str, List[float]] = {
            k: [] for k in self.stats_to_track
        }

        self._num_episodes = 0
        self._all_steps: List[int] = []
        self._all_stats: Dict[str, List[float]] = {k: [] for k in self.stats_to_track}

        self.reset()

    def step(self, env: AgentEnvWrapper):
        self._current_steps += 1
        for k in self.stats_to_track:
            if self.planner.root.belief.size() == 0:
                v = 0.0
            elif k == "policy":
                v = self.get_other_policy_accuracy(env.policies)
            elif k == "action":
                # check belief for prediction of "next" actions (i.e. the actions that)
                # were just performed
                v = self.get_other_action_accuracy(env.last_actions)
            elif k == "state":
                # check belief for state prior to the most recent action
                v = self.get_state_accuracy(self._current_state)
            elif k == "history":
                # check belief for histories before the most recent action
                v = self.get_other_history_accuracy(self._current_other_agent_histories)
            else:
                raise ValueError(f"Unknown stat to track: {k}")
            self._current_stats[k].append(v)

        for i, a in env.last_actions.items():
            self._current_other_agent_histories[i].extend(a, env.last_obs[i])
        self._current_state = env.state

    def reset(self):
        """Reset tracker."""
        self.reset_episode()
        self._num_episodes = 0
        self._all_steps = []
        self._all_stats = {k: [] for k in self.stats_to_track}

    def reset_episode(self):
        """Reset for new episode.

        Takes care of summarizing results from any previous episode.
        This should be called at the beginning of each episode, or after finishing
        final episode.
        """
        assert all(
            a is None for a in self.env.last_actions.values()
        ), "Must call reset environment before resetting BeliefStatTracker."
        self._current_other_agent_histories = {
            i: AgentHistory.get_init_history(o) for i, o in self.env.last_obs.items()
        }
        self._current_state = self.env.state

        if self._current_steps == 0:
            return

        self._num_episodes += 1
        if self.track_overall:
            self._all_steps.append(self._current_steps)
            for k, v in self._current_stats:
                self._all_stats[k].append(np.nanmean(v))

        self._current_steps = 0
        self._current_stats = {k: [] for k in self.stats_to_track}

    def get_episode(self) -> Dict[str, float]:
        """Get statistics for current episode."""
        stats = {}
        for k, v in self._current_stats.items():
            if len(v) == 0 or np.isnan(np.sum(v)):
                v_mean = np.nan
            else:
                v_mean = np.nanmean(v, axis=0)
            stats[k] = v_mean
        return stats

    def get(self) -> Dict[str, float]:
        """Get statistics for all episodes."""
        stats = {}
        for k, v in self._all_stats.items():
            if len(v) == 0 or np.isnan(np.sum(v)):
                mean_val = np.nan
                std_val = np.nan
            else:
                mean_val = np.nanmean(v, axis=0)
                std_val = np.nanstd(v, axis=0)
            stats[f"{k}_mean"] = mean_val
            stats[f"{k}_std"] = std_val
        return stats

    def get_state_accuracy(self, state: M.StateType) -> float:
        """Get state belief accuracy."""
        state_count = 0
        for hps in self.planner.root.belief.particles:
            state_count += int(hps.state == state)
        return state_count / self.planner.root.belief.size()

    def get_other_history_accuracy(self, histories: Dict[str, AgentHistory]) -> float:
        """Get belief accuracy of other agent's history."""
        h_count = 0
        for hps in self.planner.root.belief.particles:
            if hps.history is None:
                continue
            if all(hps.history.get_agent_history(i) == h for i, h in histories.items()):
                h_count += 1
        return h_count / self.planner.root.belief.size()

    def get_other_policy_accuracy(self, policies: Dict[str, pga.Policy]) -> float:
        """Get belief accuracy of other agent's policy ID/type."""
        pi_count = 0
        for hps in self.planner.root.belief.particles:
            if hps.policy_state is None:
                continue
            correct = True
            for i, pi_state in hps.policy_state.items():
                if i == self.planner.agent_id:
                    continue

                if pi_state.get("policy_id", None) != policies[i].policy_id:
                    correct = False
                    break
            pi_count += int(correct)
        return pi_count / self.planner.root.belief.size()

    def get_other_action_accuracy(self, actions: Dict[str, M.ActType]) -> float:
        """Get belief accuracy of other agent's action."""
        a_prob_sum = 0
        for hps in self.planner.root.belief.particles:
            if hps.policy_state is None:
                continue
            joint_a_prob = 1.0
            for i, a in actions.items():
                other_pi = self.planner.other_agent_policies[i].get_pi(
                    hps.policy_state[i]
                )
                joint_a_prob *= other_pi.get(a, 0.0)
            a_prob_sum += joint_a_prob
        return a_prob_sum / self.planner.root.belief.size()
