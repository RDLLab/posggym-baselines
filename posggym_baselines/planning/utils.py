from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np

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
