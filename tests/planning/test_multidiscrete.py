import posggym
import posggym.agents as pga
from posggym_baselines.planning import (
    POMCP,
    MCTSConfig,
    RandomSearchPolicy,
    load_posggym_agents_search_policy,
)
from posggym.wrappers import DiscretizeActions, DiscretizeObservations

env = DiscretizeObservations(posggym.make("DrivingContinuous-v0", render_mode="human"))
wrapped_env = DiscretizeActions(env, num_actions=10, flatten=True)

config = MCTSConfig(
    discount=0.95,           # expected return discount factor
    search_time_limit=0.1,   # per step search time
    c=1.414,                 # ~ math.sqrt(2)
    truncated=False,         # use monte-carlo rollouts
    action_selection="ucb",  # search action selection policy
)

planning_agent_id = env.possible_agents[0]
other_agent_id = env.possible_agents[1]

planner = POMCP(
    wrapped_env.model,
    planning_agent_id,
    config,
    search_policy=RandomSearchPolicy(wrapped_env.model, planning_agent_id),
)

obs, infos = env.reset()
print(obs)
all_done = False

planner.reset()
while not all_done:
    env.render()
    actions = {
        planning_agent_id: planner.step(obs[planning_agent_id]),
        other_agent_id: env.action_spaces[other_agent_id].sample()
    }
    print(actions)
    obs, rewards, terms, truncs, all_done, infos = env.step(actions)

env.close()
planner.close()