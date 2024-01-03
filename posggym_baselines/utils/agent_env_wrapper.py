"""Functions for the posggym.agents.wrappers.AgentEnvWrapper."""
from typing import Dict, List

import posggym
import posggym.agents as pga


class UniformOtherAgentFn:
    """Function for loading other agents.

    Samples the other agents from a uniform distribution over set of possible agents.

    This is a callable class that can be pickled and passed to the workers.
    """

    def __init__(self, agent_policy_ids: Dict[str, List[str]]):
        self.agent_policy_ids = agent_policy_ids
        self.policies = {i: {} for i in agent_policy_ids}

    def __call__(self, model: posggym.POSGModel) -> Dict[str, pga.Policy]:
        other_agents = {}
        for agent_id in self.agent_policy_ids:
            pi_id = model.rng.choice(self.agent_policy_ids[agent_id])
            if pi_id not in self.policies[agent_id]:
                self.policies[agent_id][pi_id] = pga.make(pi_id, model, agent_id)
            other_agents[agent_id] = self.policies[agent_id][pi_id]
        return other_agents
