from typing import Dict

import posggym.model as M

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.mcts import MCTS
from posggym_baselines.planning.other_policy import OtherAgentPolicy
from posggym_baselines.planning.search_policy import SearchPolicy


class IPOMCP(MCTS):
    """Interactive Partially Observable Monte-Carlo Planning (I-POMCP).

    This implementation varies a bit from the original I-POMCP and CI-I-POMCP algorithms
    in that it assumes knowledge of the other agents' policies, and uses rejection
    sampling instead of weighted particle filtering to update the belief state.

    The original I-POMCP algorithm used I-POMDP-Lite to find the policies of lower-level
    agents, unfortunately I-POMDP-Lite would not scale to the size of the environments
    that we are interested in.

    CI-I-POMCP extends I-POMCP to environments with explicit communication and uses
    either an offline solver for the lower level models or a nested MCTS approach. The
    nested MCTS approach is equivelent to I-NTMCP (implemented in a seperate file),
    while this implemntation is equivalent to the offline solver approach but where the
    other agent policies can be arbitrary.

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: MCTSConfig,
        other_agent_policies: Dict[str, OtherAgentPolicy],
        search_policy: SearchPolicy,
    ):
        super().__init__(model, agent_id, config, other_agent_policies, search_policy)
