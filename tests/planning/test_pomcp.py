"""Tests for planning.potmmcp.core."""
import math

import posggym
import posggym.agents as pga

from posggym_baselines.planning.mcts import POMMCPConfig
from posggym_baselines.planning.pomcp import POMCP
from posggym_baselines.planning.search_policy import (
    RandomSearchPolicy,
    load_posggym_agents_search_policy,
)


def run_policies(env, planner, other_policy, planning_agent_id, other_agent_id):
    obs, _ = env.reset()
    done = False

    planner.reset()
    other_policy.reset()

    while not done:
        planner_action = planner.step(obs[planning_agent_id])
        true_other_action = other_policy.step(obs[other_agent_id])
        actions = {planning_agent_id: planner_action, other_agent_id: true_other_action}

        obs, _, _, _, done, _ = env.step(actions)
        env.render()

    env.close()
    planner.close()
    other_policy.close()


def test_with_random_search_policy(render_mode="human"):
    """Test POMCP with random search policy."""
    config = POMMCPConfig(
        discount=0.95,
        search_time_limit=0.1,
        c=math.sqrt(2),
        truncated=False,
        action_selection="pucb",
        root_exploration_fraction=0.25,
        known_bounds=None,
        step_limit=None,
        epsilon=0.92,  # depth_limit=2
        seed=0,
        state_belief_only=True,
    )

    env = posggym.make(
        "Driving-v1",
        grid="14x14RoundAbout",
        num_agents=2,
        obs_dim=(3, 1, 1),
        render_mode=render_mode,
    )

    planning_agent_id = env.possible_agents[0]
    other_agent_id = env.possible_agents[1]

    search_policy = RandomSearchPolicy(env.model, planning_agent_id)

    planner = POMCP(
        env.model,
        planning_agent_id,
        config,
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


def test_with_pga_search_policy(render_mode="human"):
    """Test POMCP with POSGGym.Agents search policy."""
    config = POMMCPConfig(
        discount=0.95,
        search_time_limit=0.1,
        c=math.sqrt(2),
        truncated=False,
        action_selection="pucb",
        root_exploration_fraction=0.25,
        known_bounds=None,
        step_limit=None,
        epsilon=0.92,  # depth_limit=2
        seed=0,
        state_belief_only=True,
    )

    env = posggym.make(
        "Driving-v1",
        grid="14x14RoundAbout",
        num_agents=2,
        obs_dim=(3, 1, 1),
        render_mode=render_mode,
    )

    planning_agent_id = env.possible_agents[0]
    other_agent_id = env.possible_agents[1]

    search_policy = load_posggym_agents_search_policy(
        env.model, planning_agent_id, "Driving-v1/A0Shortestpath-v0"
    )

    planner = POMCP(
        env.model,
        planning_agent_id,
        config,
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


if __name__ == "__main__":
    test_with_random_search_policy("human")
    test_with_pga_search_policy("human")
