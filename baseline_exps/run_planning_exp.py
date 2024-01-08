"""Script for running POTMMCP baseline experiments.

Specifically, for a given baseline environment and population (P0, P1) and mode
(in-distribution, out-of-distribution), run POTMMCP for `num_episodes` episodes for
various search budgets and save the results to a file.


in-distribution - planning agent is evaluated against the same population that it uses
    for planning
out-of-distribution - planning agent is evaluated against different population to the
    one it uses for planning

"""
import argparse
import copy
import itertools
import multiprocessing as mp
import pprint
import time
from datetime import datetime
from typing import List
from pathlib import Path

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
        )
        all_exp_params.append(exp_params)
        exp_num += 1

    return all_exp_params


def main(args):
    print("Running experiments with the following parameters:")
    print("Env ID:", args.env_id)
    print("Agent ID:", args.agent_id)
    print("Num episodes:", args.num_episodes)
    print("Search times:", args.search_times)
    print("Exp time limit:", args.exp_time_limit)

    # limit number of threads to 1 for each process
    torch.set_num_threads(1)

    env_data = exp_utils.get_env_data(args.env_id, args.agent_id)
    print("env_kwargs:")
    pprint.pprint(env_data.env_kwargs)
    print("agents_P0:")
    pprint.pprint(env_data.agents_P0)
    print("agents_P1:")
    pprint.pprint(env_data.agents_P1)

    exp_name = f"{args.alg_name}_{args.env_id}"
    if args.agent_id is not None:
        exp_name += f"_i{args.agent_id}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_results_parent_dir = exp_utils.RESULTS_DIR / exp_name
    exp_results_parent_dir.mkdir(exist_ok=True)

    if args.alg_name == "INTMCP":
        all_exp_params = get_intmcp_exp_params(
            args, env_data, exp_name, exp_results_parent_dir
        )
    elif args.alg_name == "IPOMCP":
        all_exp_params = get_ipomcp_exp_params(
            args, env_data, exp_name, exp_results_parent_dir
        )
    elif args.alg_name == "POMCP":
        all_exp_params = get_pomcp_exp_params(
            args, env_data, exp_name, exp_results_parent_dir
        )
    elif args.alg_name == "POTMMCP":
        all_exp_params = get_potmmcp_exp_params(
            args, env_data, exp_name, exp_results_parent_dir
        )
    else:
        raise ValueError(f"Unknown algorithm name: {args.alg_name}")

    print(f"Total number of experiments={len(all_exp_params)}")

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
        "alg_name",
        type=str,
        choices=["INTMCP", "IPOMCP", "POMCP", "POTMMCP"],
        help="Planning algorithm to run.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Name of environment.",
    )
    parser.add_argument(
        "--agent_id",
        type=str,
        default=None,
        help="ID of agent.",
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
        "--n_cpus",
        type=int,
        default=1,
        help="Number of cpus to use for running experiments in parallel.",
    )
    args = parser.parse_args()

    if args.env_id == "all":
        print("Running all envs")
        for env_id in exp_utils.ENV_DATA_DIR.glob("*"):
            if not (exp_utils.ENV_DATA_DIR / env_id).is_dir():
                continue
            if env_id.endswith("_i0"):
                agent_id = "0"
                env_id = env_id.replace("_i0", "")
            elif env_id.endswith("_i1"):
                agent_id = "1"
                env_id = env_id.replace("_i1", "")
            else:
                agent_id = None
            args.env_id = env_id
            args.agent_id = agent_id
            print()
            main(args)
    else:
        if args.agent_id == "" or args.agent_id == "None":
            args.agent_id = None
        main(args)
