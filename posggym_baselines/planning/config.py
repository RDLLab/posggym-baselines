import math
from dataclasses import dataclass, field
from typing import Optional

from posggym_baselines.planning.utils import KnownBounds


@dataclass
class MCTSConfig:
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
    state_belief_only: bool = False

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
