from posggym_baselines.ppo.core import load_policies
from copy import deepcopy
from typing import Callable, Optional

from baseline_exps import exp_utils
import posggym
from posggym.wrappers import FlattenObservations, RecordVideo, DiscretizeActions

from posggym_baselines.ppo.config import PPOConfig
from posggym_baselines.ppo.eval import render_policies
from posggym_baselines.ppo.ippo import IPPOConfig
from posggym_baselines.ppo.klr_ppo import KLRPPOConfig
from gymnasium import spaces
from typing_extensions import Annotated
import typer
from enum import Enum
from pathlib import Path
import re
import functools
from matplotlib import pyplot as plt

app = typer.Typer()


class TrainableEnvs(str, Enum):
    CooperativeReaching = "CooperativeReaching-v0"
    Driving = "Driving-v1"
    DrivingContinuous = "DrivingContinuous-v0"
    LevelBasedForaging = "LevelBasedForaging-v3"
    PredatorPrey = "PredatorPrey-v0"
    PursuitEvasion_i0 = "PursuitEvasion-v1_i0"
    PursuitEvasion_i1 = "PursuitEvasion-v1_i1"


class Algs(str, Enum):
    SP = "SP"
    SP_BR = "SP-BR"
    KLR = "KLR"
    KLR_BR = "KLR-BR"


def get_env_creator_fn(
    config: PPOConfig,
    env_idx: int,
    worker_idx: Optional[int] = None,
    n_actions: int = 4,
) -> Callable:
    """Get function for creating the environment."""

    def thunk():
        capture_video = config.capture_video and env_idx == 0 and worker_idx == 0
        env = posggym.make(config.env_id, render_mode="human", **config.env_kwargs)
        if capture_video:
            env = RecordVideo(env, config.video_dir)
        env = FlattenObservations(env)

        if all(
            isinstance(env.action_spaces[key], spaces.Box) for key in env.action_spaces
        ):
            env = DiscretizeActions(env, n_actions, False)

        # seed = random.random
        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx

        env.reset(seed=seed)
        return env

    return thunk


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    ctx: typer.Context,
    pop_training_alg: Annotated[Algs, typer.Option(case_sensitive=False)],
    full_env_id: Annotated[TrainableEnvs, typer.Option()],
    pop_size: Annotated[
        int,
        typer.Option(
            help="Number of independent policies in the population."
            "For `klr` and `klr-br`, this is the max reasoning"
            "level `K`, e.g `pop_size=4` means `K=3` for  `klr` and"
            " `k=2` for `klr-br`."
        ),
    ] = 4,
    track_wandb: Annotated[bool, typer.Option()] = True,
    disable_logging: Annotated[bool, typer.Option()] = False,
    capture_video: Annotated[bool, typer.Option()] = False,
    cuda: Annotated[bool, typer.Option()] = True,
    seed: Annotated[int, typer.Option()] = 0,
    total_timesteps: Annotated[int, typer.Option()] = int(3.2e6),
    num_workers: Annotated[int, typer.Option()] = 2,
    use_lstm: Annotated[bool, typer.Option()] = True,
    checkpoint: Annotated[Optional[int], typer.Option()] = None,
    log_dir: Annotated[Path, typer.Option()] = Path("."),
    load_dir: Annotated[Optional[Path], typer.Option()] = None,
    n_actions: Annotated[int, typer.Option()] = 4,
):
    d = deepcopy(locals())

    """Run training for KLR and KLR plus BR."""
    env_data = exp_utils.get_env_data(None, None, full_env_id=full_env_id.value)
    env_kwargs = env_data.env_kwargs

    env_data_path = exp_utils.ENV_DATA_DIR / full_env_id.value
    if (env_data_path / "PPOConfig.yaml").exists():
        import yaml

        with open(env_data_path / "PPOConfig.yaml", "r") as file:
            loaded_config = yaml.safe_load(file)
    else:
        loaded_config = {}
    # env_kwargs["env_kwargs"]["render_mode"] = "human"
    config_kwargs = deepcopy(exp_utils.DEFAULT_PPO_CONFIG)
    config_kwargs.update(
        {
            "exp_name": f"{pop_training_alg.value}_{full_env_id.value}_N{pop_size}",
            "env_creator_fn": functools.partial(
                get_env_creator_fn, n_actions=n_actions
            ),
            "env_id": env_kwargs["env_id"],
            "env_kwargs": env_kwargs["env_kwargs"],
        }
    )
    config_kwargs = {**config_kwargs, **loaded_config}

    for k, v in d.items():
        if k in config_kwargs:
            if isinstance(v, Enum):
                config_kwargs[k] = v.value
            else:
                config_kwargs[k] = v

    for arg in ctx.args:
        pattern = r"--(\w+)=(\w+)"
        matches = re.match(pattern, arg)
        # print(arg, matches)

        if matches:
            key = matches.group(1)
            value = matches.group(2)
            if key in config_kwargs["env_kwargs"]:
                original_type = type(config_kwargs["env_kwargs"][key])
                config_kwargs["env_kwargs"][key] = original_type(value)
            elif key in config_kwargs:
                original_type = type(config_kwargs[key])
                config_kwargs[key] = original_type(value)
    # return
    if pop_training_alg.value.startswith("KLR"):
        if pop_training_alg == Algs.KLR_BR:
            include_BR = True
            max_reasoning_level = pop_size - 2
        else:
            include_BR = False
            max_reasoning_level = pop_size - 1

        config = KLRPPOConfig(
            max_reasoning_level=max_reasoning_level,
            include_BR=include_BR,
            filter_experience=True,
            **config_kwargs,
        )
    else:
        if pop_training_alg == Algs.SP_BR:
            include_BR = True
            pop_size = pop_size - 1
        else:
            include_BR = False
            pop_size = pop_size

        config = IPPOConfig(
            pop_size=pop_size,
            include_BR=include_BR,
            filter_experience=True,
            **config_kwargs,
        )
    if load_dir is not None:
        print("checkpoint=", checkpoint)
        p = load_policies(
            config, checkpoint=checkpoint, save_dir=load_dir, device=config.eval_device
        )

    env = config.load_vec_env(1)
    mass_results, friction_results, elasticity_results, result = render_policies(
        [p, p], 1000, env, config, render=False
    )
    p1, p2 = list(zip(*result))

    # Separate data into true and false categories
    true_data = [point[1:] for point in p1 if point[0]]
    false_data = [point[1:] for point in p1 if not point[0]]

    # Plot the data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    import pdb

    pdb.set_trace()

    # True data points in green
    for point in true_data:
        ax.scatter(point[0], point[1], point[2], color="green")

    # False data points in red
    for point in false_data:
        ax.scatter(point[0], point[1], point[2], color="red")

    # Setting labels
    ax.set_xlabel("Mass")
    ax.set_ylabel("Friction")
    ax.set_zlabel("Elasticitiy")
    with open("out_of_distribution.txt", "w") as f:
        for x, y in result:
            f.write(str(x) + "," + str(y))

    plt.show()

    # import pdb; pdb.set_trace()
    # print(x)
    # print("anc")


if __name__ == "__main__":
    app()
    # command = typer.main.get_command(app)
    # import sys

    # try:
    #    result = command(standalone_mode=False)
    #    sys.exit(result)
    # except:
    #    print(f'exception was thrown')
