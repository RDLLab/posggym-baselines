"""Script for compiling planning experiment results.

The experiment for each planning algorithm will produce a directory containing a
results directory for each (planning pop, testing pop, search time) sub-experiment. Each
results directory contains `episode_results.csv` and `exp_args.yaml` files. This script
combines the results from all sub-experiments into a single `planning_results.csv` file.
It may also be used to combine results from multiple experiments (i.e. multiple
algorithms and/or environments) into a single file.

Script has two modes:

1. `single`, will compile results for single experiments (i.e. a single (alg, env)
    combination), into a single `planning_results.csv` file.
2. `all`, will compile results for all experiments (alg, env) in a parent directory into
    a single `planning_results.csv` file. This is equivalent to running `single` mode
    for each (alg, env) combination in the parent directory, then combining the results
    into a single file.

"""
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from exp_utils import ENV_DATA_DIR


def compile_sub_experiment_results(
    parent_dir: Path, save_to_file: bool = True, summarize: bool = True
) -> pd.DataFrame:
    """Compile results from all sub-experiments of (alg, env) into a single file.

    Arguments
    ---------
    parent_dir
        Path to parent directory containing experiment results
    save_to_file
        Whether to save compiled results to file. Will be saved to `parent_dir` as
        `planning_results.csv`.
    summarize
        Whether to summarize results over all episodes. If True, will compute mean,
        std, and 95% confidence interval over all episodes. Otherwise will return
        results for each episode.

    Returns
    -------
    combined_results
        Dataframe containing combined results from all sub-experiments.

    """
    print(f"Compiling results from all sub-experiments in {parent_dir}")
    parent_dir = Path(parent_dir)

    # get all sub-experiment results directories
    results_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    # get summary results file for each sub-experiment
    all_sub_exp_results = []
    for results_dir in results_dirs:
        results_file = results_dir / "episode_results.csv"
        exp_args_file = results_dir / "exp_args.yaml"

        sub_exp_results_df = pd.read_csv(results_file)
        if summarize:
            # get the mean, std, 95CI over all episodes
            num_episodes = sub_exp_results_df.shape[0]
            assert num_episodes == sub_exp_results_df["num"].max() + 1
            mean_results = sub_exp_results_df.mean(axis=0).to_dict()
            std_results = sub_exp_results_df.std(axis=0).to_dict()
            summary_results = {
                "num_episodes": num_episodes,
            }
            for k, v in mean_results.items():
                if k == "num":
                    continue
                summary_results[f"{k}_mean"] = [v]
                summary_results[f"{k}_std"] = [std_results[k]]
                summary_results[f"{k}_95ci"] = [
                    1.96 * std_results[k] / (num_episodes**0.5)
                ]

            sub_exp_results_df = pd.DataFrame(summary_results)

        # add in experiment arguments
        with open(exp_args_file, "r") as f:
            exp_args = yaml.safe_load(f)

        for k, v in exp_args.items():
            if k == "num_episodes":
                k = "num_episodes_limit"
            sub_exp_results_df.insert(0, k, v)

        # add experiment arguments to summarized results
        all_sub_exp_results.append(sub_exp_results_df)

    # combine all episode results into a single dataframe
    combined_results = pd.concat(all_sub_exp_results, ignore_index=True)

    # save combined results
    if save_to_file:
        print(f"Saving results to {parent_dir / 'planning_results.csv'}")
        combined_results.to_csv(parent_dir / "planning_results.csv", index=False)

    return combined_results


def combine_all_experiment_results(
    parent_dir: Path,
    save_to_file: bool = True,
    summarize: bool = True,
    save_to_main_results_dir: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Combine results from all experiments (alg, env) in `parent_dir`.

    Assumes each (alg, env) is in its own sub-directory of `parent_dir` with name
    `<alg>_<env_id>[_agent_id]_<date>_<time>` where `agent_id` is optional.


    Arguments
    ---------
    parent_dirs
        List of paths to parent directories containing experiment results.
    save_to_file
        Whether to save combined results to file. Will be saved to `parent_dir` as
        `<env_id>[_agent_id]_planning_results.csv`.
    summarize
        Whether to summarize results over all episodes. If True, will compute mean,
        std, and 95% confidence interval over all episodes for each (alg, env)
        sub-experiment Otherwise results will contain results for each episode.

    Returns
    -------
    combined_results
        Dataframe for each environment containing combined results from all algorithms.

    """
    print(f"Compiling results from all (env, alg) experiments in {parent_dir}")
    parent_dir = Path(parent_dir)

    # get all sub-experiment results directories
    exp_parent_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    # maps env_id -> [alg results]
    all_env_exp_results = {}
    for exp_parent_dir in exp_parent_dirs:
        tokens = exp_parent_dir.name.split("_")
        alg, env_id = tokens[:2]
        if tokens[2].startswith("i"):
            env_id += "_" + tokens[2]

        if env_id not in all_env_exp_results:
            all_env_exp_results[env_id] = []

        env_alg_results = compile_sub_experiment_results(
            exp_parent_dir, save_to_file=False, summarize=summarize
        )

        # add alg name as column to results
        env_alg_results.insert(0, "alg", alg)
        all_env_exp_results[env_id].append(env_alg_results)

    combined_results = {
        env_id: pd.concat(all_env_exp_results[env_id], ignore_index=True)
        for env_id in all_env_exp_results
    }

    # save combined results
    if save_to_file:
        if summarize:
            save_file_suffix = "planning_summary_results.csv"
        else:
            save_file_suffix = "planning_results.csv"

        for env_id in combined_results:
            if save_to_main_results_dir:
                env_save_file = ENV_DATA_DIR / env_id / save_file_suffix
            else:
                env_save_file = parent_dir / f"{env_id}_{save_file_suffix}"
            print(f"Saving results to {env_save_file}")
            combined_results[env_id].to_csv(env_save_file, index=False)

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["single", "all"],
        help="Compilation mode",
    )

    parser.add_argument(
        "exp_results_parent_dirs",
        type=Path,
        nargs="+",
        help="Path to parent directory (or multiple) containing experiment results.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Whether to summarize results across episodes",
    )
    parser.add_argument(
        "--save_to_main_results_dir",
        action="store_true",
        help=(
            "Whether to save results to main `env_data` dir. Will overwrite existing "
            "results so be careful."
        ),
    )
    args = parser.parse_args()

    for parent_dir in args.exp_results_parent_dirs:
        if args.mode == "single":
            compile_sub_experiment_results(parent_dir, summarize=args.summarize)
        elif args.mode == "all":
            combine_all_experiment_results(
                parent_dir,
                summarize=args.summarize,
                save_to_main_results_dir=args.save_to_main_results_dir,
            )
