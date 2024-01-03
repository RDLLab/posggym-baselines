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
import csv
import itertools
import multiprocessing as mp
import os
import pprint
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import exp_utils
import posggym
import torch
import yaml
from posggym.agents.wrappers import AgentEnvWrapper

from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.other_policy import OtherAgentMixturePolicy
from posggym_baselines.planning.potmmcp import POTMMCP, POTMMCPMetaPolicy
from posggym_baselines.utils.agent_env_wrapper import UniformOtherAgentFn

DEFAULT_SEARCH_TIMES = [0.1, 1.0, 5.0, 10.0, 20.0]
DEFAULT_NUM_EPISODES = 500
DEFAULT_EXP_TIME_LIMIT = 60 * 60 * 48  # 48 hours

# experiments limited to softmax meta policy since it's generally the best
DEFAULT_META_POLICY = "softmax"

DEFAULT_CONFIG_KWARGS = {
    "discount": 0.99,  # same as RL experiments
    # "search_time_limit": 0.1,   # Set below
    "c": 1.25,
    "truncated": True,
    "action_selection": "pucb",
    "root_exploration_fraction": 0.5,
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": None,  # Set in algorithm, if env has a step limit
    "epsilon": 0.01,
    "seed": None,
    "state_belief_only": False,
    # fallback to rollout if search policy has no value function
    "use_rollout_if_no_value": True,
}


@dataclass
class POTMMCPExpParams:
    """Parameters for running POTMMCP experiments."""

    # stuff used for running experiments
    env_kwargs: Dict
    agent_id: Optional[str]
    config: MCTSConfig
    meta_policy: Dict[str, Dict[str, float]]
    # other agent policies used by planning agent (i.e. agent has knowledge of)
    planning_other_agent_policy_ids: Dict[str, List[str]]
    # other agent policies that planning agent is evaluated against
    test_other_agent_policy_ids: Dict[str, List[str]]
    # number of episodes
    num_episodes: int
    # time limit for experiment
    exp_time_limit: int

    # experiment details for saving results
    exp_name: str
    exp_num: int
    exp_results_parent_dir: str
    planning_pop_id: str
    test_pop_id: str

    exp_results_dir: str = field(init=False)
    episode_results_file: str = field(init=False)
    exp_args_file: str = field(init=False)
    exp_log_file: str = field(init=False)
    episode_results_heads: List[str] = field(init=False)
    exp_start_time: float = field(init=False)

    def __post_init__(self):
        search_time = self.config.search_time_limit
        if search_time < 1.0:
            search_time_str = f"{search_time:.1f}"
        else:
            search_time_str = f"{search_time:.0f}"

        self.exp_results_dir = os.path.join(
            self.exp_results_parent_dir,
            f"{self.exp_num}_{self.planning_pop_id}_{self.test_pop_id}_{search_time_str}",
        )

        self.episode_results_file = os.path.join(
            self.exp_results_dir, "episode_results.csv"
        )
        self.exp_args_file = os.path.join(self.exp_results_dir, "exp_args.yaml")
        self.exp_log_file = os.path.join(self.exp_results_dir, "exp_log.txt")

        self.episode_results_heads = [
            "num",
            "len",
            "return",
            "discounted_return",
            "time",
        ]

    def setup_exp(self):
        """Runs necessary setup for experiment.

        I.e. setup results directy, files, logging, etc
        """
        self.exp_start_time = time.time()

        if not os.path.exists(self.exp_results_dir):
            os.makedirs(self.exp_results_dir)

        exp_args = {
            "exp_num": self.exp_num,
            "env_id": self.env_kwargs["env_id"],
            "agent_id": self.agent_id,
            "planning_pop_id": self.planning_pop_id,
            "test_pop_id": self.test_pop_id,
            "search_time_limit": self.config.search_time_limit,
            "num_episodes": self.num_episodes,
            "exp_time_limit": self.exp_time_limit,
        }
        with open(self.exp_args_file, "w") as f:
            yaml.safe_dump(exp_args, f)

        with open(self.episode_results_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_results_heads)
            writer.writeheader()

        with open(self.exp_log_file, "w") as f:
            f.write(f"Experiment Name: {self.exp_name}\n")
            f.write(f"Experiment Num: {self.exp_num}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Experiment Args:\n")
            f.write(pprint.pformat(exp_args, indent=4, sort_dicts=False) + "\n\n")

    def finalize_exp(self):
        """Runs necessary finalization for experiment.

        I.e. save results, etc
        """
        time_taken = time.time() - self.exp_start_time
        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)

        with open(self.exp_log_file, "a") as f:
            f.write("Experiment finished" + "\n")
            f.write(f"Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time Taken={hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}\n")

    def write_episode_results(self, results: Dict):
        with open(self.episode_results_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_results_heads)
            writer.writerow(results)

    def write_log(self, log: str, add_timestamp: bool = True):
        if add_timestamp:
            time_taken = time.time() - self.exp_start_time
            hours, rem = divmod(time_taken, 3600)
            minutes, seconds = divmod(rem, 60)
            log = f"[{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}] {log}"

        with open(self.exp_log_file, "a") as f:
            f.write(log + "\n")


def run_exp(exp_params: POTMMCPExpParams):
    print(f"Running experiment {exp_params.exp_num}")
    exp_params.setup_exp()

    # initialize environment (including folding in test population)
    env = posggym.make(
        exp_params.env_kwargs["env_id"], **exp_params.env_kwargs["env_kwargs"]
    )
    other_agent_fn = UniformOtherAgentFn(exp_params.test_other_agent_policy_ids)
    env = AgentEnvWrapper(env, other_agent_fn)

    # initialize planning agent
    search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
        env.model, exp_params.agent_id, exp_params.meta_policy
    )
    planner_other_agent_policies = {
        i: OtherAgentMixturePolicy.load_posggym_agents_policy(env.model, i, policy_ids)
        for i, policy_ids in exp_params.planning_other_agent_policy_ids.items()
    }
    planner = POTMMCP(
        env.model,
        exp_params.agent_id,
        config=exp_params.config,
        other_agent_policies=planner_other_agent_policies,
        search_policy=search_policy,
    )

    # run episode loop
    exp_start_time = time.time()
    episode_num = 0
    while (
        episode_num < exp_params.num_episodes
        and time.time() - exp_start_time < exp_params.exp_time_limit
    ):
        obs, _ = env.reset()
        planner.reset()

        episode_results = {
            "num": episode_num,
            "len": 0,
            "return": 0,
            "discounted_return": 0,
            "time": 0,
        }
        episode_start_time = time.time()
        done = False
        while not done:
            action = planner.step(obs[exp_params.agent_id])
            obs, rewards, terms, truncs, all_done, _ = env.step(
                {exp_params.agent_id: action}
            )

            # only care about planning agent finishing
            done = terms[exp_params.agent_id] or truncs[exp_params.agent_id] or all_done

            reward = rewards[exp_params.agent_id]
            episode_results["return"] += reward
            episode_results["discounted_return"] += (
                exp_params.config.discount ** episode_results["len"] * reward
            )
            episode_results["len"] += 1

        episode_results["time"] = time.time() - episode_start_time
        exp_params.write_episode_results(episode_results)
        episode_num += 1

        if episode_num % max(1, exp_params.num_episodes // 10) == 0:
            exp_params.write_log(
                f"Episode {episode_num}/{exp_params.num_episodes} complete.",
                add_timestamp=True,
            )

    # finalize experiment
    env.close()
    planner.close()
    exp_params.finalize_exp()
    print(f"Experiment {exp_params.exp_num} complete.")


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

    exp_name = f"POTMMCP_{args.env_id}"
    if args.agent_id is not None:
        exp_name += f"_i{args.agent_id}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_results_parent_dir = os.path.join(exp_utils.RESULTS_DIR, exp_name)
    if not os.path.exists(exp_results_parent_dir):
        os.makedirs(exp_results_parent_dir)

    # generate all experiment parameters
    all_exp_params = []
    exp_num = 0
    for planning_pop_id, test_pop_id, search_time in itertools.product(
        ["P0", "P1"], ["P0", "P1"], args.search_times
    ):
        exp_params = POTMMCPExpParams(
            env_kwargs=env_data.env_kwargs,
            agent_id=env_data.agent_id,
            config=MCTSConfig(search_time_limit=search_time, **DEFAULT_CONFIG_KWARGS),
            meta_policy=copy.deepcopy(
                env_data.meta_policy[planning_pop_id][DEFAULT_META_POLICY]
            ),
            planning_other_agent_policy_ids=(
                env_data.agents_P0 if planning_pop_id == "P0" else env_data.agents_P1
            ),
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

    print(f"Total number of experiments={len(all_exp_params)}")

    start_time = time.time()
    # run experiments
    with mp.Pool(args.n_cpus) as pool:
        pool.map(run_exp, all_exp_params)

    time.sleep(1)
    time_taken = time.time() - start_time
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total time taken={hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--search_times",
        type=float,
        nargs="+",
        default=DEFAULT_SEARCH_TIMES,
        help="Search times to evaluate.",
    )
    parser.add_argument(
        "--exp_time_limit",
        type=int,
        default=DEFAULT_EXP_TIME_LIMIT,
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
        for env_id in os.listdir(exp_utils.ENV_DATA_DIR):
            if env_id == "Driving-v1":
                continue
            if env_id.endswith("_i0"):
                agent_id = "i0"
                env_id = env_id.replace("_i0", "")
            elif env_id.endswith("_i1"):
                agent_id = "i1"
                env_id = env_id.replace("_i1", "")
            else:
                agent_id = None
            args.env_id = env_id
            args.agent_id = agent_id
            print()
            main(args)
    else:
        main(args)
