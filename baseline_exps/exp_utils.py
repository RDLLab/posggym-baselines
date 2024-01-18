import csv
import math
import os
import pprint
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import posggym
import psutil
import yaml
from posggym.agents.wrappers import AgentEnvWrapper
from posggym_baselines.planning.config import MCTSConfig
from posggym_baselines.planning.utils import PlanningStatTracker
from posggym_baselines.utils.agent_env_wrapper import UniformOtherAgentFn


BASELINE_EXP_DIR = Path(__file__).resolve().parent
ENV_DATA_DIR = BASELINE_EXP_DIR / "env_data"
RESULTS_DIR = BASELINE_EXP_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Default parameters
DEFAULT_SEARCH_TIMES = [0.1, 1.0, 5.0, 10.0, 20.0]
DEFAULT_NUM_EPISODES = 500
DEFAULT_EXP_TIME_LIMIT = 60 * 60 * 48  # 48 hours

DEFAULT_PLANNING_CONFIG_KWARGS_PUCB = {
    "discount": 0.99,
    # "search_time_limit": 0.1,   # Set below
    "c": 1.25,
    "truncated": True,
    "action_selection": "pucb",
    "pucb_exploration_fraction": 0.5,
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": None,  # Set in algorithm, if env has a step limit
    "epsilon": 0.01,
    "seed": None,
    "state_belief_only": False,
    # fallback to rollout if search policy has no value function
    "use_rollout_if_no_value": True,
}


DEFAULT_PLANNING_CONFIG_KWARGS_UCB = dict(DEFAULT_PLANNING_CONFIG_KWARGS_PUCB)
DEFAULT_PLANNING_CONFIG_KWARGS_UCB["action_selection"] = "ucb"
DEFAULT_PLANNING_CONFIG_KWARGS_UCB["truncated"] = False
DEFAULT_PLANNING_CONFIG_KWARGS_UCB["c"] = math.sqrt(2)


DEFAULT_PPO_CONFIG = {
    "eval_fns": [],
    # general config
    # "exp_name": "br_ppo",  # added later
    "seed": 0,
    "cuda": True,
    "torch_deterministic": False,
    "worker_device": "cpu",
    "disable_logging": False,
    "track_wandb": True,
    "wandb_project": "posggym_baselines",
    "wandb_entity": None,
    "wandb_group": None,
    "load_dir": None,
    "save_interval": -1,
    # env config
    "capture_video": False,
    # eval configuration
    "eval_interval": 100,
    "eval_episodes": 100,
    "num_eval_envs": 32,
    # training config
    "total_timesteps": int(1e8),
    "num_workers": 2,
    "num_envs": 32,
    "num_rollout_steps": 64,
    # batch size = 32 * 64 * 2 = 4096
    "seq_len": 10,
    "minibatch_size": 2048,
    # minibatch_num_seqs roughly = 2048 // 10 = 205
    "update_epochs": 2,
    # PPO update hyperparams
    "learning_rate": 3e-4,
    "anneal_lr": False,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 10,
    "target_kl": None,
    # model config
    "use_lstm": True,
    "lstm_size": 64,
    "lstm_num_layers": 1,
    "trunk_sizes": [64, 64],
    "head_sizes": [64],
}


PURSUITEVASION_POLICY_NAMES = {
    0: {
        "P0": [f"KLR{i}_i0" for i in list(range(5)) + ["BR"]],
        "P1": [f"RL{i+1}_i0" for i in range(6)],
    },
    1: {
        "P0": [f"KLR{i}_i1" for i in list(range(5)) + ["BR"]],
        "P1": [f"RL{i+1}_i1" for i in range(6)],
    },
}

PURSUITEVASION_POLICY_NAMES_TO_IDS = {}
for agent_id, pop_map in PURSUITEVASION_POLICY_NAMES.items():
    for pop_policy_names in pop_map.values():
        for policy_name in pop_policy_names:
            PURSUITEVASION_POLICY_NAMES_TO_IDS[
                policy_name
            ] = f"PursuitEvasion-v1/grid=16x16/{policy_name}-v0"


@dataclass
class EnvData:
    """Data for a particular environment."""

    # Environment data
    full_env_id: str
    env_id: str
    agent_id: str
    other_agent_id: str

    # env_kwargs contains `env_id` and `env_kwargs` keys
    env_kwargs: Dict[str, Dict[str, Any]]
    env_data_dir: Path

    # Population data
    # agent_id -> List[policy_id]
    agents_P0: Dict[str, List[str]]
    agents_P1: Dict[str, List[str]]
    pop_div_results_file: Path
    # policy names are shorthand for policy IDs
    # pop_id -> List[str]
    pop_policy_names: Dict[str, List[str]]
    pop_co_team_names: Dict[str, List[str]]
    # map from policy_name -> policy_id
    policy_name_to_id: Dict[str, str]

    # RL data
    # [P0, P1] -> [seed] -> model_file
    br_model_files: Dict[str, Dict[int, Path]]
    rl_br_results_file: Path
    rl_br_training_results_files: Dict[str, Path]

    # Planning data
    # Meta policy pop_id -> meta_policy_type -> meta_policy
    # [P0, P1] -> [greedy, softmax, uniform] -> meta_policy
    meta_policy: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
    # Experiment results
    planning_results_file: Path
    planning_summary_results_file: Path
    combined_results_file: Path

    def pprint(self, exlude_large_entries: bool = True):
        output = [f"EnvData ({self.full_env_id}):"]
        for field_name, field_value in self.__dict__.items():
            if exlude_large_entries and field_name in ["meta_policy", "br_model_files"]:
                continue
            formated_value = pprint.pformat(field_value, depth=2, indent=4)
            output.append(f"{field_name}={formated_value}")
        print("\n  ".join(output))


# TODO remove env_id and agent_id and use full_env_id only
#      (do this after all experiments are done)
def get_env_data(
    env_id: Optional[str], agent_id: Optional[str], full_env_id: Optional[str] = None
):
    """Get the env data for the given env name."""
    if full_env_id is None:
        assert env_id is not None
        full_env_id = env_id if agent_id is None else f"{env_id}_i{agent_id}"
    else:
        if len(full_env_id.split("_")) == 2:
            env_id, agent_id = full_env_id.split("_")
            agent_id = agent_id.replace("i", "")
        else:
            env_id = full_env_id
            agent_id = None
    env_data_path = ENV_DATA_DIR / full_env_id

    env_kwargs_file = env_data_path / "env_kwargs.yaml"
    with open(env_kwargs_file, "r") as f:
        env_kwargs = yaml.safe_load(f)

    # Population Data
    agents_P0_file = env_data_path / (
        "agents_P0.yaml" if agent_id is None else f"agents_P0_i{agent_id}.yaml"
    )

    with open(agents_P0_file, "r") as f:
        agents_P0 = yaml.safe_load(f)

    agents_P1_file = env_data_path / (
        "agents_P1.yaml" if agent_id is None else f"agents_P1_i{agent_id}.yaml"
    )

    with open(agents_P1_file, "r") as f:
        agents_P1 = yaml.safe_load(f)

    other_agent_id = next(iter(agents_P0.keys()))
    assert other_agent_id == next(iter(agents_P1.keys()))
    ego_agent_id = "0" if other_agent_id == "1" else "1"

    pop_policy_names = {}
    pop_co_team_names = {}
    policy_name_to_id = {}
    if full_env_id == "PursuitEvasion-v1_i0":
        for pop_id in ["P0", "P1"]:
            pop_policy_names[pop_id] = PURSUITEVASION_POLICY_NAMES[1][pop_id]
            pop_co_team_names[pop_id] = PURSUITEVASION_POLICY_NAMES[0][pop_id]
        policy_name_to_id = PURSUITEVASION_POLICY_NAMES_TO_IDS
    elif full_env_id == "PursuitEvasion-v1_i1":
        for pop_id in ["P0", "P1"]:
            pop_policy_names[pop_id] = PURSUITEVASION_POLICY_NAMES[0][pop_id]
            pop_co_team_names[pop_id] = PURSUITEVASION_POLICY_NAMES[1][pop_id]
        policy_name_to_id = PURSUITEVASION_POLICY_NAMES_TO_IDS
    else:
        for pop_id, pop in zip(["P0", "P1"], [agents_P0, agents_P1]):
            policy_ids = list(pop.values())[0]
            policy_names = []
            for policy_id in policy_ids:
                policy_name = policy_id.split("/")[-1].split("-v")[0]
                policy_names.append(policy_name)
                policy_name_to_id[policy_name] = policy_id

            pop_policy_names[pop_id] = policy_names
            pop_co_team_names[pop_id] = policy_names

    # RL Data
    br_models_dir = env_data_path / "br_models"
    br_model_files = {"P0": {}, "P1": {}}
    for model_file_name in br_models_dir.glob("*.pt"):
        model_name = model_file_name.with_suffix("").name
        tokens = model_name.split("_")
        train_pop = tokens[0]
        if agent_id is not None:
            assert tokens[1].startswith("i")
            assert tokens[1] == f"i{agent_id}"
            seed = int(tokens[2].replace("seed", ""))
        else:
            seed = int(tokens[1].replace("seed", ""))
        br_model_files[train_pop][seed] = br_models_dir / model_file_name

    # Planninn Data
    meta_policy_file = env_data_path / "meta_policy.yaml"
    with open(meta_policy_file, "r") as f:
        meta_policy = yaml.safe_load(f)

    return EnvData(
        full_env_id=full_env_id,
        env_id=env_id,
        agent_id=ego_agent_id,
        other_agent_id=other_agent_id,
        env_kwargs=env_kwargs,
        env_data_dir=env_data_path,
        agents_P0=agents_P0,
        agents_P1=agents_P1,
        pop_div_results_file=env_data_path / "div_results.csv",
        pop_policy_names=pop_policy_names,
        pop_co_team_names=pop_co_team_names,
        policy_name_to_id=policy_name_to_id,
        br_model_files=br_model_files,
        rl_br_results_file=env_data_path / "br_results.csv",
        rl_br_training_results_files={
            p: env_data_path / f"{p}_br_training_results.csv" for p in ["P0", "P1"]
        },
        meta_policy=meta_policy,
        planning_results_file=env_data_path / "planning_results.csv",
        planning_summary_results_file=env_data_path / "planning_summary_results.csv",
        combined_results_file=env_data_path / "combined_results.csv",
    )


def load_all_env_data() -> Dict[str, EnvData]:
    """Load data for all environments."""
    all_env_data = {}
    full_env_ids = sorted([f.name for f in ENV_DATA_DIR.glob("*")])
    for full_env_id in full_env_ids:
        if not (ENV_DATA_DIR / full_env_id).is_dir():
            continue
        all_env_data[full_env_id] = get_env_data(None, None, full_env_id=full_env_id)
    return all_env_data


@dataclass
class PlanningExpParams:
    """Parameters for running planning experiments."""

    # stuff used for running experiments
    env_kwargs: Dict
    agent_id: Optional[str]
    config: MCTSConfig
    # kwargs and fn for initializing planner
    planner_init_fn: Callable[[posggym.POSGModel, "PlanningExpParams"], Any]
    planner_kwargs: Dict
    # other agent policies that planning agent is evaluated against
    test_other_agent_policy_ids: Dict[str, List[str]]
    # number of episodes
    num_episodes: int
    # time limit for experiment
    exp_time_limit: int

    # experiment details for saving results
    exp_name: str
    exp_num: int
    exp_results_parent_dir: Path
    planning_pop_id: str
    test_pop_id: str
    full_env_id: str

    exp_results_dir: Path = field(init=False)
    episode_results_file: str = field(init=False)
    exp_args_file: Path = field(init=False)
    exp_log_file: Path = field(init=False)
    episode_results_heads: List[str] = field(init=False)
    exp_start_time: float = field(init=False)

    def __post_init__(self):
        self.exp_results_dir = (
            self.exp_results_parent_dir / self.get_exp_results_file_name()
        )

        self.episode_results_file = self.exp_results_dir / "episode_results.csv"
        self.exp_args_file = self.exp_results_dir / "exp_args.yaml"
        self.exp_log_file = self.exp_results_dir / "exp_log.txt"

        self.episode_results_heads = [
            "num",
            "len",
            "return",
            "discounted_return",
            "time",
        ] + PlanningStatTracker.STAT_KEYS

    def get_exp_results_file_name(self):
        file_name = f"{self.exp_num}_{self.planning_pop_id}_{self.test_pop_id}"
        file_name += f"_{self.config.search_time_limit:1g}"
        return file_name

    def get_exp_args(self):
        return {
            "exp_num": self.exp_num,
            "env_id": self.env_kwargs["env_id"],
            "agent_id": self.agent_id,
            "planning_pop_id": self.planning_pop_id,
            "test_pop_id": self.test_pop_id,
            "search_time_limit": self.config.search_time_limit,
            "num_episodes": self.num_episodes,
            "exp_time_limit": self.exp_time_limit,
        }

    def setup_exp(self):
        """Runs necessary setup for experiment.

        I.e. setup results directly, files, logging, etc
        """
        self.exp_start_time = time.time()

        self.exp_results_dir.mkdir(exist_ok=True)

        exp_args = self.get_exp_args()
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

    def write_log(
        self,
        log: str,
        add_timestamp: bool = True,
        report_mem_usage: bool = True,
    ):
        if report_mem_usage:
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss / 1024**2
            vrt = process.memory_info().vms / 1024**2
            shr = process.memory_info().shared / 1024**2
            log = f"(MEM={vrt:.0f}/{rss:.0f}/{shr:.0f}MB VIRT/RSS/SHR) {log}"

        if add_timestamp:
            time_taken = time.time() - self.exp_start_time
            hours, rem = divmod(time_taken, 3600)
            minutes, seconds = divmod(rem, 60)
            log = f"[{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}] {log}"

        with open(self.exp_log_file, "a") as f:
            f.write(log + "\n")


@dataclass
class CombinedExpParams(PlanningExpParams):
    """Parameters for running combined planning + RL experiments."""

    # RL policy stuff
    rl_policy_seed: int
    rl_policy_pop_id: str
    env_data_dir: Path

    rl_policy_file: Path = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        policy_file_name = f"{self.rl_policy_pop_id}"
        if self.full_env_id == "PursuitEvasion-v1_i0":
            policy_file_name += "_i0"
        elif self.full_env_id == "PursuitEvasion-v1_i1":
            policy_file_name += "_i1"
        policy_file_name += f"_seed{self.rl_policy_seed}.pt"

        self.rl_policy_file = self.env_data_dir / "br_models" / policy_file_name

    def get_exp_results_file_name(self):
        file_name = f"{self.exp_num}_{self.planning_pop_id}_{self.test_pop_id}"
        file_name += f"_rl{self.rl_policy_pop_id}_s{self.rl_policy_seed}"
        file_name += f"_{self.config.search_time_limit:1g}"
        return file_name

    def get_exp_args(self):
        return {
            "exp_num": self.exp_num,
            "env_id": self.env_kwargs["env_id"],
            "agent_id": self.agent_id,
            "planning_pop_id": self.planning_pop_id,
            "test_pop_id": self.test_pop_id,
            "rl_policy_pop_id": self.rl_policy_pop_id,
            "rl_policy_seed": self.rl_policy_seed,
            "search_time_limit": self.config.search_time_limit,
            "num_episodes": self.num_episodes,
            "exp_time_limit": self.exp_time_limit,
        }


def run_planning_exp(exp_params: PlanningExpParams):
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Running experiment {exp_params.exp_name} {exp_params.exp_num}: "
        f"{exp_params.planning_pop_id} vs {exp_params.test_pop_id} with "
        f"search_time={exp_params.config.search_time_limit}."
    )
    exp_params.setup_exp()

    # initialize environment (including folding in test population)
    env = posggym.make(
        exp_params.env_kwargs["env_id"], **exp_params.env_kwargs["env_kwargs"]
    )
    other_agent_fn = UniformOtherAgentFn(exp_params.test_other_agent_policy_ids)
    env = AgentEnvWrapper(env, other_agent_fn)

    planner = exp_params.planner_init_fn(env.model, exp_params)
    # disable tracking overall stats since we only log per episode
    planner.stat_tracker.track_overall = False

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
        episode_results.update(planner.stat_tracker.get_episode())
        exp_params.write_episode_results(episode_results)
        episode_num += 1

        if episode_num % max(1, exp_params.num_episodes // 100) == 0:
            exp_params.write_log(
                f"Episode {episode_num}/{exp_params.num_episodes} complete.",
                add_timestamp=True,
                report_mem_usage=True,
            )

    # finalize experiment
    env.close()
    planner.close()
    exp_params.finalize_exp()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Experiment {exp_params.exp_name} {exp_params.exp_num} complete."
    )
