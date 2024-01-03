"""Tests for planning.potmmcp.core."""
import math

import posggym
import posggym.agents as pga
from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.other_policy import OtherAgentMixturePolicy
from posggym_baselines.planning.potmmcp import POTMMCP, POTMMCPMetaPolicy


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


def test_with_single_random_policies():
    """Test POMMCP with random policies."""
    config = MCTSConfig(
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

    meta_policy = {
        "Random-v0": {"Random-v0": 1.0},
    }

    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        env.model, planning_agent_id, meta_policy
    )
    planner_other_policy = OtherAgentMixturePolicy.load_posggym_agents_policy(
        env.model, other_agent_id, ["Random-v0"]
    )

    planner = POTMMCP(
        env.model,
        planning_agent_id,
        config,
        other_agent_policies={other_agent_id: planner_other_policy},
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


def test_with_other_policies_and_random_meta_policy():
    """Test POMMCP with random policies."""
    config = MCTSConfig(
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

    other_agent_policy_ids = [
        "Driving-v1/A0Shortestpath-v0",
        "Driving-v1/A40Shortestpath-v0",
        "Driving-v1/A60Shortestpath-v0",
        "Driving-v1/A80Shortestpath-v0",
        "Driving-v1/A100Shortestpath-v0",
    ]

    meta_policy = {pi_id: {"Random-v0": 1.0} for pi_id in other_agent_policy_ids}

    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        env.model, planning_agent_id, meta_policy
    )
    planner_other_policy = OtherAgentMixturePolicy.load_posggym_agents_policy(
        env.model, other_agent_id, other_agent_policy_ids
    )

    planner = POTMMCP(
        env.model,
        planning_agent_id,
        config,
        other_agent_policies={other_agent_id: planner_other_policy},
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


def test_with_other_policies_and_uniform_meta_policy():
    """Test POMMCP with random policies."""
    config = MCTSConfig(
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

    other_agent_policy_ids = [
        "Driving-v1/A0Shortestpath-v0",
        "Driving-v1/A40Shortestpath-v0",
        "Driving-v1/A60Shortestpath-v0",
        "Driving-v1/A80Shortestpath-v0",
        "Driving-v1/A100Shortestpath-v0",
    ]

    meta_policy = {
        pi_id_j: {
            pi_id_i: 1.0 / len(other_agent_policy_ids)
            for pi_id_i in other_agent_policy_ids
        }
        for pi_id_j in other_agent_policy_ids
    }

    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        env.model, planning_agent_id, meta_policy
    )
    planner_other_policy = OtherAgentMixturePolicy.load_posggym_agents_policy(
        env.model, other_agent_id, other_agent_policy_ids
    )

    planner = POTMMCP(
        env.model,
        planning_agent_id,
        config,
        other_agent_policies={other_agent_id: planner_other_policy},
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


def test_with_torch_other_and_meta_policies():
    """Test POMMCP with random policies."""
    config = MCTSConfig(
        discount=0.95,
        search_time_limit=0.1,
        c=math.sqrt(2),
        truncated=True,
        action_selection="pucb",
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

    other_agent_policy_ids = [
        "Driving-v1/grid=14x14RoundAbout-num_agents=2/RL1-v0",
        "Driving-v1/grid=14x14RoundAbout-num_agents=2/RL2-v0",
        "Driving-v1/grid=14x14RoundAbout-num_agents=2/RL3-v0",
        "Driving-v1/grid=14x14RoundAbout-num_agents=2/RL4-v0",
        "Driving-v1/grid=14x14RoundAbout-num_agents=2/RL5-v0",
    ]

    meta_policy = {
        pi_id_j: {
            pi_id_i: 1.0 / len(other_agent_policy_ids)
            for pi_id_i in other_agent_policy_ids
        }
        for pi_id_j in other_agent_policy_ids
    }

    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        env.model, planning_agent_id, meta_policy
    )
    planner_other_policy = OtherAgentMixturePolicy.load_posggym_agents_policy(
        env.model, other_agent_id, other_agent_policy_ids
    )

    planner = POTMMCP(
        env.model,
        planning_agent_id,
        config,
        other_agent_policies={other_agent_id: planner_other_policy},
        search_policy=search_policy,
    )

    true_other_policy = pga.make("Random-v0", env.model, other_agent_id)

    run_policies(env, planner, true_other_policy, planning_agent_id, other_agent_id)


if __name__ == "__main__":
    # test_with_single_random_policies()
    # test_with_other_policies_and_random_meta_policy()
    # test_with_other_policies_and_uniform_meta_policy()
    test_with_torch_other_and_meta_policies()
