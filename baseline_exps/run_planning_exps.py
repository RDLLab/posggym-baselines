"""Script for running planning baseline experiments.

For given planning algorithm and baseline environment the script runs the planning
algorithm for `num_episodes` episodes for selected `search_times` and saves the results
to a file. Experiments are run both population `P0` and `P1` as the planning population
and tested  against both populations, leading to 4 * |`search_times`| experiments.

The script can be used to run multiple planning algorithms and multiple environments.
Furthermore individual experiments can be run in parallel using multiple cpus (see
`n_cpus` argument).

Available algorithms:

- INTMCP
- IPOMCP
- POMCP
- POTMMCP

Available environments:

- CooperativeReaching-v0
- Driving-v1
- LevelBasedForaging-v3
- PredatorPrey-v0
- PursuitEvasion-v1_i0
- PursuitEvasion-v1_i1

Examples:

    # run INTMCP on Driving-v1 for 100 episodes for 1s and 10s search times
    python run_planning_exps.py \
        --alg_names INTMCP \
        --full_env_ids Driving-v1 \
        --search_times 1 10

    # run all algorithms on all environments for 100 episodes with 10 cpus
    python run_planning_exps.py \
        --alg_names all \
        --full_env_ids all \
        --search_times 1 10 \
        --num_episodes 100 \
        --n_cpus 10

"""
import argparse
import copy
import itertools
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import List

import exp_utils
import posggym
import torch
from exp_utils import PlanningExpParams

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.intmcp import INTMCP
from posggym_baselines.planning.ipomcp import IPOMCP
from posggym_baselines.planning.other_policy import OtherAgentMixturePolicy
from posggym_baselines.planning.pomcp import POMCP
from posggym_baselines.planning.potmmcp import POTMMCP, POTMMCPMetaPolicy
from posggym_baselines.planning.search_policy import RandomSearchPolicy
from posggym_baselines.utils import strtobool

# same as in I-POMCP paper experiments
# also best performing value in I-NTMCP paper
# also around what humans use
DEFAULT_INTMCP_NESTING_LEVEL = 2

# experiments limited to softmax meta policy since it's generally the best
DEFAULT_META_POLICY = "softmax"


def init_intmcp(model: posggym.POSGModel, exp_params: PlanningExpParams) -> INTMCP:
    planner = INTMCP.initialize(
        model,
        exp_params.agent_id,
        config=exp_params.config,
        nesting_level=exp_params.planner_kwargs["nesting_level"],
        search_policies=None,  # Use RandomSearchPolicy
    )
    return planner


def get_intmcp_exp_params(
    args, env_data: exp_utils.EnvData, exp_name: str, exp_results_parent_dir: Path
) -> List[PlanningExpParams]:
    config_kwargs = dict(exp_utils.DEFAULT_PLANNING_CONFIG_KWARGS_UCB)
    config_kwargs["truncated"] = False

    belief_stats_to_track = []
    if args.track_belief_stats:
        belief_stats_to_track = ["state", "history"]

    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time in itertools.product(
        ["P0", "P1"], ["P0", "P1"], args.search_times
    ):
        exp_params = PlanningExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(search_time_limit=search_time, **config_kwargs),
            planner_init_fn=init_intmcp,
            planner_kwargs={
                "nesting_level": args.nesting_level,
            },
            test_other_agent_policy_ids=(
                env_data.agents_P0 if test_pop_id == "P0" else env_data.agents_P1
            ),
            num_episodes=args.num_episodes,
            exp_time_limit=args.exp_time_limit,
            exp_name=exp_name,
            exp_num=exp_num,
            exp_results_parent_dir=exp_results_parent_dir,
            planning_pop_id=planning_pop_id,
            test_pop_id=test_pop_id,
            full_env_id=env_data.full_env_id,
            belief_stats_to_track=[*belief_stats_to_track],
        )
        all_exp_params.append(exp_params)
        exp_num += 1

    return all_exp_params


def init_ipomcp(model: posggym.POSGModel, exp_params: PlanningExpParams) -> IPOMCP:
    search_policy = RandomSearchPolicy(model, exp_params.agent_id)
    planner_other_agent_policies = {
        i: OtherAgentMixturePolicy.load_posggym_agents_policy(model, i, policy_ids)
        for i, policy_ids in exp_params.planner_kwargs[
            "planning_other_agent_policy_ids"
        ].items()
    }
    planner = IPOMCP(
        model,
        exp_params.agent_id,
        config=exp_params.config,
        other_agent_policies=planner_other_agent_policies,
        search_policy=search_policy,
    )
    return planner


def get_ipomcp_exp_params(
    args, env_data: exp_utils.EnvData, exp_name: str, exp_results_parent_dir: Path
) -> List[PlanningExpParams]:
    config_kwargs = dict(exp_utils.DEFAULT_PLANNING_CONFIG_KWARGS_UCB)
    config_kwargs["truncated"] = False

    belief_stats_to_track = []
    if args.track_belief_stats:
        belief_stats_to_track = ["state", "history", "action", "policy"]

    # generate all experiment parameters
    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time in itertools.product(
        ["P0", "P1"], ["P0", "P1"], args.search_times
    ):
        exp_params = PlanningExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(search_time_limit=search_time, **config_kwargs),
            planner_init_fn=init_ipomcp,
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
            num_episodes=args.num_episodes,
            exp_time_limit=args.exp_time_limit,
            exp_name=exp_name,
            exp_num=exp_num,
            exp_results_parent_dir=exp_results_parent_dir,
            planning_pop_id=planning_pop_id,
            test_pop_id=test_pop_id,
            full_env_id=env_data.full_env_id,
            belief_stats_to_track=[*belief_stats_to_track],
        )
        all_exp_params.append(exp_params)
        exp_num += 1
    return all_exp_params


def init_pomcp(model: posggym.POSGModel, exp_params: PlanningExpParams) -> POMCP:
    search_policy = RandomSearchPolicy(model, exp_params.agent_id)
    planner = POMCP(
        model,
        exp_params.agent_id,
        config=exp_params.config,
        search_policy=search_policy,
    )
    return planner


def get_pomcp_exp_params(
    args, env_data: exp_utils.EnvData, exp_name: str, exp_results_parent_dir: Path
) -> List[PlanningExpParams]:
    config_kwargs = dict(exp_utils.DEFAULT_PLANNING_CONFIG_KWARGS_UCB)
    config_kwargs["truncated"] = False
    config_kwargs["state_belief_only"] = True

    belief_stats_to_track = []
    if args.track_belief_stats:
        belief_stats_to_track = ["state"]

    # generate all experiment parameters
    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time in itertools.product(
        ["P0", "P1"], ["P0", "P1"], args.search_times
    ):
        exp_params = PlanningExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(search_time_limit=search_time, **config_kwargs),
            planner_init_fn=init_pomcp,
            planner_kwargs={},
            test_other_agent_policy_ids=(
                env_data.agents_P0 if test_pop_id == "P0" else env_data.agents_P1
            ),
            num_episodes=args.num_episodes,
            exp_time_limit=args.exp_time_limit,
            exp_name=exp_name,
            exp_num=exp_num,
            exp_results_parent_dir=exp_results_parent_dir,
            planning_pop_id=planning_pop_id,
            test_pop_id=test_pop_id,
            full_env_id=env_data.full_env_id,
            belief_stats_to_track=[*belief_stats_to_track],
        )
        all_exp_params.append(exp_params)
        exp_num += 1

    return all_exp_params


def init_potmmcp(model: posggym.POSGModel, exp_params: PlanningExpParams) -> POTMMCP:
    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        model, exp_params.agent_id, exp_params.planner_kwargs["meta_policy"]
    )
    planner_other_agent_policies = {
        i: OtherAgentMixturePolicy.load_posggym_agents_policy(model, i, policy_ids)
        for i, policy_ids in exp_params.planner_kwargs[
            "planning_other_agent_policy_ids"
        ].items()
    }
    planner = POTMMCP(
        model,
        exp_params.agent_id,
        config=exp_params.config,
        other_agent_policies=planner_other_agent_policies,
        search_policy=search_policy,
    )
    return planner


def get_potmmcp_exp_params(
    args, env_data: exp_utils.EnvData, exp_name: str, exp_results_parent_dir: Path
) -> List[PlanningExpParams]:
    belief_stats_to_track = []
    if args.track_belief_stats:
        belief_stats_to_track = ["state", "history", "action", "policy"]

    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time in itertools.product(
        ["P0", "P1"], ["P0", "P1"], args.search_times
    ):
        exp_params = PlanningExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(
                search_time_limit=search_time,
                **exp_utils.DEFAULT_PLANNING_CONFIG_KWARGS_PUCB,
            ),
            planner_init_fn=init_potmmcp,
            planner_kwargs={
                "meta_policy": copy.deepcopy(
                    env_data.meta_policy[planning_pop_id][DEFAULT_META_POLICY]
                ),
                "planning_other_agent_policy_ids": (
                    env_data.agents_P0
                    if planning_pop_id == "P0"
                    else env_data.agents_P1
                ),
            },
            test_other_agent_policy_ids=(
                env_data.agents_P0 if test_pop_id == "P0" else env_data.agents_P1
            ),
            num_episodes=args.num_episodes,
            exp_time_limit=args.exp_time_limit,
            exp_name=exp_name,
            exp_num=exp_num,
            exp_results_parent_dir=exp_results_parent_dir,
            planning_pop_id=planning_pop_id,
            test_pop_id=test_pop_id,
            full_env_id=env_data.full_env_id,
            belief_stats_to_track=[*belief_stats_to_track],
        )
        all_exp_params.append(exp_params)
        exp_num += 1

    return all_exp_params


def main(args):
    print("Running experiments with the following parameters:")
    print("Algorithms:", args.alg_names)
    print("Env IDs:", args.full_env_ids)
    print("Nesting level:", args.nesting_level)
    print("Num episodes:", args.num_episodes)
    print("Search times:", args.search_times)
    print("Exp time limit:", args.exp_time_limit)
    print("N cpus:", args.n_cpus)

    # limit number of threads to 1 for each process
    torch.set_num_threads(1)

    if args.alg_names == ["all"]:
        args.alg_names = ["INTMCP", "IPOMCP", "POMCP", "POTMMCP"]
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
    for alg_name, full_env_id in itertools.product(args.alg_names, args.full_env_ids):
        tokens = full_env_id.split("_")
        if len(tokens) == 1:
            env_id = full_env_id
            agent_id = None
        elif len(tokens) == 2:
            env_id = tokens[0]
            agent_id = tokens[1].replace("i", "")
        else:
            raise ValueError("Invalid full_env_id: {}".format(full_env_id))

        env_data = exp_utils.get_env_data(env_id, agent_id)
        exp_name = f"{alg_name}_{env_id}"
        if agent_id is not None:
            exp_name += f"_i{agent_id}"
        exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        exp_results_parent_dir = exp_utils.RESULTS_DIR / exp_name
        exp_results_parent_dir.mkdir(exist_ok=True)

        if alg_name == "INTMCP":
            exp_params = get_intmcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "IPOMCP":
            exp_params = get_ipomcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "POMCP":
            exp_params = get_pomcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "POTMMCP":
            exp_params = get_potmmcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        else:
            raise ValueError(f"Unknown algorithm name: {alg_name}")
        all_exp_params.extend(exp_params)

    print(f"Total number of experiments={len(all_exp_params)}")

    start_time = time.time()
    # run experiments
    with mp.Pool(args.n_cpus, maxtasksperchild=1) as pool:
        print("Running experiments...")
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
        "--alg_names",
        type=str,
        nargs="+",
        required=True,
        choices=["INTMCP", "IPOMCP", "POMCP", "POTMMCP", "all"],
        help="Planning algorithm to run.",
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
        help="Name of environment.",
    )
    parser.add_argument(
        "--nesting_level",
        type=int,
        default=DEFAULT_INTMCP_NESTING_LEVEL,
        help="Nesting level to use for I-NTMCP planner.",
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
        "--track_belief_stats",
        type=strtobool,
        default=False,
        help="Whether to track belief accuracy statistics.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=1,
        help="Number of cpus to use for running experiments in parallel.",
    )
    args = parser.parse_args()
    main(args)
