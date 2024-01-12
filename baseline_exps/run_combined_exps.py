"""Script for running combined planning+RL baseline experiments.

For a given environment runs the planning algorithm for  `num_episodes` episodes for
selected `search_times` save the results to a file.
Runs for both population `P0` and `P1` and tests against both  populations, leading to
4 * |`search_times`| experiments.

Available environments:

- CooperativeReaching-v0
- Driving-v1
- LevelBasedForaging-v3
- PredatorPrey-v0
- PursuitEvasion-v1_i0
- PursuitEvasion-v1_i1

Examples:

    # run on Driving-v1 for 100 episodes for 1s and 10s
    python run_combined_exp.py --full_env_id Driving-v1 --search_times 1 10

"""
import argparse
import itertools
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import List

import exp_utils
import posggym.model as M
import torch
from exp_utils import CombinedExpParams
from posggym.agents.utils import processors

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.mcts import MCTS
from posggym_baselines.planning.other_policy import OtherAgentMixturePolicy
from posggym_baselines.planning.search_policy import PPOLSTMSearchPolicy
from posggym_baselines.ppo.network import PPOLSTMModel

# Number of different seeds used to train RL policies
NUM_RL_POLICY_SEEDS = 5


def init_combined(model: M.POSGModel, exp_params: CombinedExpParams) -> MCTS:
    checkpoint = torch.load(exp_params.rl_policy_file, map_location="cpu")
    config = checkpoint["config"]

    act_space = model.action_spaces[exp_params.agent_id]
    obs_space = model.observation_spaces[exp_params.agent_id]
    obs_processor = processors.FlattenProcessor(obs_space)
    flattened_obs_space = obs_processor.get_processed_space()

    rl_policy = PPOLSTMModel(
        input_size=flattened_obs_space.shape[0],
        num_actions=act_space.n,
        trunk_sizes=config["trunk_sizes"],
        lstm_size=config["lstm_size"],
        lstm_layers=config["lstm_num_layers"],
        head_sizes=config["head_sizes"],
    )
    rl_policy.to("cpu")
    rl_policy.eval()
    rl_policy.requires_grad_(False)
    rl_policy.load_state_dict(checkpoint["model"])

    search_policy = PPOLSTMSearchPolicy(
        model,
        exp_params.agent_id,
        f"BR_{exp_params.rl_policy_pop_id}_seed{exp_params.rl_policy_seed}",
        rl_policy,
        obs_processor,
    )

    planner_other_agent_policies = {
        i: OtherAgentMixturePolicy.load_posggym_agents_policy(model, i, policy_ids)
        for i, policy_ids in exp_params.planner_kwargs[
            "planning_other_agent_policy_ids"
        ].items()
    }
    planner = MCTS(
        model,
        exp_params.agent_id,
        config=exp_params.config,
        other_agent_policies=planner_other_agent_policies,
        search_policy=search_policy,
    )
    return planner


def get_combined_exp_params(
    env_data: exp_utils.EnvData,
    exp_name: str,
    exp_results_parent_dir: Path,
    search_times: List[float],
    num_episodes: int,
    exp_time_limit: int,
) -> List[CombinedExpParams]:
    config_kwargs = dict(exp_utils.DEFAULT_PLANNING_CONFIG_KWARGS_PUCB)

    # generate all experiment parameters
    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time, rl_seed in itertools.product(
        ["P0", "P1"], ["P0", "P1"], search_times, range(NUM_RL_POLICY_SEEDS)
    ):
        exp_params = CombinedExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(search_time_limit=search_time, **config_kwargs),
            planner_init_fn=init_combined,
            planner_kwargs={
                "planning_other_agent_policy_ids": (
                    env_data.agents_P0
                    if planning_pop_id == "P0"
                    else env_data.agents_P1
                ),
            },
            test_other_agent_policy_ids=(
                env_data.agents_P0 if test_pop_id == "P0" else env_data.agents_P1
            ),
            num_episodes=num_episodes,
            exp_time_limit=exp_time_limit,
            exp_name=exp_name,
            exp_num=exp_num,
            exp_results_parent_dir=exp_results_parent_dir,
            planning_pop_id=planning_pop_id,
            test_pop_id=test_pop_id,
            full_env_id=env_data.full_env_id,
            rl_policy_seed=rl_seed,
            rl_policy_pop_id=planning_pop_id,
            env_data_dir=env_data.env_data_dir,
        )
        all_exp_params.append(exp_params)
        exp_num += 1
    return all_exp_params


def main(args):
    print("Running experiments with the following parameters:")
    print("Full Env ID:", args.full_env_ids)
    print("Num episodes:", args.num_episodes)
    print("Search times:", args.search_times)
    print("Exp time limit:", args.exp_time_limit)
    print("N cpus:", args.n_cpus)

    # limit number of threads to 1 for each process
    torch.set_num_threads(1)

    if "all" in args.full_env_ids:
        args.full_env_ids = [
            "CooperativeReaching-v0",
            "Driving-v1",
            "LevelBasedForaging-v3",
            "PredatorPrey-v0",
            "PursuitEvasion-v1_i0",
            "PursuitEvasion-v1_i1",
        ]

    all_exp_params = []
    for full_env_id in args.full_env_ids:
        env_data = exp_utils.get_env_data(None, None, full_env_id)
        exp_name = f"COMBINED_{full_env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        exp_results_parent_dir = exp_utils.RESULTS_DIR / exp_name
        exp_results_parent_dir.mkdir(exist_ok=True)

        exp_params = get_combined_exp_params(
            env_data,
            exp_name,
            exp_results_parent_dir,
            args.search_times,
            args.num_episodes,
            args.exp_time_limit,
        )
        all_exp_params.extend(exp_params)

    print(f"Total number of experiments={len(all_exp_params)}")

    # sort experiments by search time, longest first (can help with scheduling)
    all_exp_params.sort(key=lambda x: x.config.search_time_limit, reverse=True)

    start_time = time.time()
    # run experiments
    if args.n_cpus == 1:
        for exp_params in all_exp_params:
            exp_utils.run_planning_exp(exp_params)
    else:
        with mp.Pool(args.n_cpus, maxtasksperchild=1) as pool:
            pool.map(exp_utils.run_planning_exp, all_exp_params)

    time_taken = time.time() - start_time
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total time taken={hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--full_env_ids",
        type=str,
        nargs="+",
        required=True,
        choices=[
            "CooperativeReaching-v0",
            "Driving-v1",
            "LevelBasedForaging-v3",
            "PredatorPrey-v0",
            "PursuitEvasion-v1_i0",
            "PursuitEvasion-v1_i1",
            "all",
        ],
        help="Name of environments to run.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=exp_utils.DEFAULT_NUM_EPISODES,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--search_times",
        type=float,
        nargs="+",
        default=exp_utils.DEFAULT_SEARCH_TIMES,
        help="Search times to evaluate.",
    )
    parser.add_argument(
        "--exp_time_limit",
        type=int,
        default=exp_utils.DEFAULT_EXP_TIME_LIMIT,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=1,
        help="Number of cpus to use for running experiments in parallel.",
    )
    main(parser.parse_args())
