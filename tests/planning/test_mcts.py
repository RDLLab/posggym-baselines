"""Tests for planning.mcts.core."""
import math

import posggym
import posggym.agents as pga

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.mcts import MCTS
from posggym_baselines.planning.other_policy import RandomOtherAgentPolicy
from posggym_baselines.planning.search_policy import RandomSearchPolicy


def test_with_random_policies():
    """Test POMMCP with random policies."""
    config = MCTSConfig(
        discount=0.95,
        search_time_limit=0.1,
        c=math.sqrt(2),
        truncated=False,
        action_selection="ucb",
        root_exploration_fraction=0.25,
        known_bounds=None,
        step_limit=None,
        epsilon=0.92,  # depth_limit=2
        seed=0,
        state_belief_only=False,
    )

    env = posggym.make(
        "Driving-v1",
        grid="14x14RoundAbout",
        num_agents=2,
        obs_dim=(3, 1, 1),
        render_mode="human",
    )

    planning_agent_id = env.possible_agents[0]
    other_agent_id = env.possible_agents[1]

    search_policy = RandomSearchPolicy(env.model, planning_agent_id)
    planner_other_policy = RandomOtherAgentPolicy(env.model, other_agent_id)

    planner = MCTS(
        env.model,
        planning_agent_id,
        config,
        other_agent_policies={other_agent_id: planner_other_policy},
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    obs, _ = env.reset()
    done = False

    planner.reset()
    true_other_policy.reset()

    while not done:
        planner_action = planner.step(obs[planning_agent_id])
        true_other_action = true_other_policy.step(obs[other_agent_id])
        actions = {planning_agent_id: planner_action, other_agent_id: true_other_action}

        obs, _, _, _, done, _ = env.step(actions)
        env.render()

    env.close()
    planner.close()
    true_other_policy.close()


if __name__ == "__main__":
    test_with_random_policies()
