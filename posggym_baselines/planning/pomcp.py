import posggym.model as M

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.mcts import MCTS
from posggym_baselines.planning.other_policy import RandomOtherAgentPolicy
from posggym_baselines.planning.search_policy import SearchPolicy


class POMCP(MCTS):
    """Partially Observable Monte-Carlo Planning (POMCP).

    This implementation which is designed to work in multi-agent environments is the
    same as the Multi-Agent variant (POMMCP) that it inherits from except that the other
    agents in the environment are treated as Random agents, and the planning agent
    maintains a belief over environment state only (as opposed to environment and other
    agent state and joint history).

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        config: MCTSConfig,
        search_policy: SearchPolicy,
    ):
        other_agent_policies = {
            i: RandomOtherAgentPolicy(model, i)
            for i in model.possible_agents
            if i != agent_id
        }
        if not config.state_belief_only:
            # creates copy of config with state_belief_only=True
            config = config.replace(state_belief_only=True)
        super().__init__(model, agent_id, config, other_agent_policies, search_policy)
