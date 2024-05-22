"""Script for converting a saved model to the format used by POSGGym."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

import posggym.agents as pga


def format_checkpoint(checkpoint):
    weights = checkpoint["model"]
    config = checkpoint["config"]

    if "trunk_size" in config:
        config["trunk_sizes"] = config["trunk_size"]

    if "head_size" in config:
        config["head_sizes"] = config["head_size"]

    return pga.torch_policy.PPOTorchModelSaveFileFormat(
        weights=weights,
        trunk_sizes=config["trunk_sizes"],
        lstm_size=config["lstm_size"],
        lstm_layers=config["lstm_num_layers"],
        head_sizes=config["head_sizes"],
        activation="tanh",
        lstm_use_prev_action=config.get("lstm_use_prev_action", False),
        lstm_use_prev_reward=config.get("lstm_use_prev_reward", False),
    )


def format_checkpoint_dir(checkpoint_dir: Path, output_dir: Path | None = None):
    """Converts a directory of saved models to the POSGGym format."""
    if output_dir is None:
        output_dir = checkpoint_dir

    checkpoint_files = list(checkpoint_dir.glob("checkpoint*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    # Sort by checkpoint number, and get latest one
    checkpoint_files = sorted(checkpoint_files, key=lambda x: x.name)
    checkpoint_num = int(checkpoint_files[-1].name.split("_")[1])

    policy_checkpoint_files = {}
    for f in checkpoint_files:
        tokens = f.name.split("_")
        if tokens[1] != str(checkpoint_num):
            continue
        policy_id = "_".join(tokens[2:-1] + tokens[-1].split(".")[:1])
        policy_checkpoint_files[policy_id] = f

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    assert output_dir.is_dir()
    for policy_id, checkpoint_file in policy_checkpoint_files.items():
        save_path = output_dir / f"{policy_id}.pkl"

        checkpoint = torch.load(checkpoint_file)
        formatted_checkpoint = format_checkpoint(checkpoint)
        with open(save_path, "wb") as f:
            pickle.dump(formatted_checkpoint._asdict(), f)

        print(f"{policy_id} done")

    print(f"All done - saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save reformatted models too. Defaults to checkpoint_dir",
    )
    args = parser.parse_args()
    print(type(args.checkpoint_dir))
    format_checkpoint_dir(args.checkpoint_dir, args.output_dir)
