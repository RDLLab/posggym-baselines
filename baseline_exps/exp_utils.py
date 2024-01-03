import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


ENV_DATA_DIR = os.path.join(os.path.dirname(__file__), "env_data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


@dataclass
class EnvData:
    """Data for a particular environment."""

    # Environment data
    env_id: str
    agent_id: str
    other_agent_id: str
    # env_kwargs contains `env_id` and `env_kwargs` keys
    env_kwargs: Dict[str, Dict[str, Any]]
    env_data_dir: str

    # Population data
    # agent_id -> List[policy_id]
    agents_P0: Dict[str, List[str]]
    agents_P1: Dict[str, List[str]]
    pop_div_results_file: str

    # RL data
    # [P0, P1] -> [seed] -> model_file
    br_model_files: Dict[str, Dict[int, str]]
    rl_br_results_file: str

    # Planning data
    # Meta policy pop_id -> meta_policy_type -> meta_policy
    # [P0, P1] -> [greedy, softmax, uniform] -> meta_policy
    meta_policy: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]


def get_env_data(env_id: str, agent_id: Optional[str]):
    """Get the env data for the given env name."""
    if agent_id is None:
        env_data_path = os.path.join(ENV_DATA_DIR, env_id)
    else:
        env_data_path = os.path.join(ENV_DATA_DIR, f"{env_id}_i{agent_id}")

    env_kwargs_file = os.path.join(env_data_path, "env_kwargs.yaml")
    with open(env_kwargs_file, "r") as f:
        env_kwargs = yaml.safe_load(f)

    # Population Data
    agents_P0_file = os.path.join(
        env_data_path,
        "agents_P0.yaml" if agent_id is None else f"agents_P0_i{agent_id}.yaml",
    )
    with open(agents_P0_file, "r") as f:
        agents_P0 = yaml.safe_load(f)

    agents_P1_file = os.path.join(
        env_data_path,
        "agents_P1.yaml" if agent_id is None else f"agents_P1_i{agent_id}.yaml",
    )
    with open(agents_P1_file, "r") as f:
        agents_P1 = yaml.safe_load(f)

    other_agent_id = next(iter(agents_P0.keys()))
    assert other_agent_id == next(iter(agents_P1.keys()))

    ego_agent_id = "0" if other_agent_id == "1" else "1"

    # RL Data
    br_models_dir = os.path.join(env_data_path, "br_models")
    br_model_files = {"P0": {}, "P1": {}}
    for model_file_name in os.listdir(br_models_dir):
        model_name = model_file_name.replace(".pt", "")
        tokens = model_name.split("_")
        train_pop = tokens[0]
        if agent_id is not None:
            assert tokens[1].startswith("i")
            assert tokens[1] == f"i{agent_id}"
            seed = int(tokens[2].replace("seed", ""))
        else:
            seed = int(tokens[1].replace("seed", ""))
        br_model_files[train_pop][seed] = os.path.join(br_models_dir, model_file_name)

    # Planninn Data
    meta_policy_file = os.path.join(env_data_path, "meta_policy.yaml")
    with open(meta_policy_file, "r") as f:
        meta_policy = yaml.safe_load(f)

    return EnvData(
        env_id=env_id,
        agent_id=ego_agent_id,
        other_agent_id=other_agent_id,
        env_kwargs=env_kwargs,
        env_data_dir=env_data_path,
        agents_P0=agents_P0,
        agents_P1=agents_P1,
        pop_div_results_file=os.path.join(env_data_path, "div_results.csv"),
        br_model_files=br_model_files,
        rl_br_results_file=os.path.join(env_data_path, "br_results.csv"),
        meta_policy=meta_policy,
    )
