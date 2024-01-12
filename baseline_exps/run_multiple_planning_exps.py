"""Utility script for launching multiple independent planning experiments.

Each experiment is run as a separate subprocess.
"""
import argparse
import itertools
import multiprocessing as mp
import time
from datetime import datetime

import exp_utils
import run_planning_exp as planning_exps
import torch


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
            exp_params = planning_exps.get_intmcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "IPOMCP":
            exp_params = planning_exps.get_ipomcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "POMCP":
            exp_params = planning_exps.get_pomcp_exp_params(
                args, env_data, exp_name, exp_results_parent_dir
            )
        elif alg_name == "POTMMCP":
            exp_params = planning_exps.get_potmmcp_exp_params(
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
        default=planning_exps.DEFAULT_INTMCP_NESTING_LEVEL,
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
        "--n_cpus",
        type=int,
        default=1,
        help="Number of cpus to use for running experiments in parallel.",
    )
    args = parser.parse_args()
    main(args)
