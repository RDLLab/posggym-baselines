import argparse
import shutil
from pathlib import Path


def main(args):
    model_parent_dir: Path = args.model_parent_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(exist_ok=True)

    env_id = None
    added_models = set()
    old_to_new_paths = {}
    for model_dir_path in model_parent_dir.glob("*"):
        if not model_dir_path.is_dir():
            continue

        # format: <exp_name>_<pop_id>[_<agent_id>]_<env_id>_<seed>_<date>_<time>
        tokens = model_dir_path.name.split("_")

        pop_idx = [i for i, token in enumerate(tokens) if token in ("P0", "P1")][0]
        pop_id = tokens[pop_idx]
        if tokens[pop_idx + 1].startswith("i"):
            agent_id = tokens[pop_idx + 1]
            env_idx = pop_idx + 2
        else:
            agent_id = None
            env_idx = pop_idx + 1

        if args.agent_id is not None:
            assert agent_id is not None
            if agent_id[1:] != args.agent_id:
                print(f"Skipping {model_dir_path.name} since wrong agent ID")
                continue

        if env_id is None:
            env_id = tokens[env_idx]
        else:
            assert env_id == tokens[env_idx]

        seed = int(tokens[env_idx + 1])

        checkpoint_names = model_dir_path.glob("*BR.pt")

        assert len(checkpoint_names) >= 1
        checkpoint_path = max(checkpoint_names, key=lambda x: x.name)

        if agent_id is None:
            new_name = f"{pop_id}_seed{seed}.pt"
        else:
            new_name = f"{pop_id}_{agent_id}_seed{seed}.pt"

        assert new_name not in added_models
        added_models.add(new_name)
        new_path = output_dir / new_name
        old_to_new_paths[checkpoint_path] = new_path

    for checkpoint_path, new_path in old_to_new_paths.items():
        print(f"Copying {checkpoint_path} to {new_path}")
        shutil.copyfile(checkpoint_path, new_path)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_parent_dir",
        type=Path,
        help="Parent directory containing model directories to be extracted.",
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Directory to save extracted models."
    )
    parser.add_argument(
        "--agent_id", type=str, default=None, help="Agent ID to extract."
    )
    main(parser.parse_args())
