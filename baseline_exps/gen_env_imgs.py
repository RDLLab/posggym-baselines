import argparse
import re
from pathlib import Path

import exp_utils
import posggym
from PIL import Image


DOCS_DIR = Path(__file__).resolve().parent.parent

# snake to camel case:
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to perform in env before saving image
STEPS = 50
# height of PNG in pixels, width will be scaled to ensure correct aspec ratio
HEIGHT = 300


def gen_img(
    full_env_id: str,
    resize: bool = False,
):
    """Gen image for env."""
    env_data = exp_utils.get_env_data(full_env_id)
    env = posggym.make(
        env_data.env_id, render_mode="rgb_array", **env_data.env_kwargs["env_kwargs"]
    )

    v_file_path = exp_utils.RESULTS_DIR / (full_env_id + ".pdf")
    # obtain and save STEPS frames worth of steps
    frames = []
    while True:
        env.reset()
        done = False
        while not done and len(frames) <= STEPS:
            rgb_frame = env.render()  # type: ignore
            frames.append(Image.fromarray(rgb_frame))
            action = {i: env.action_spaces[i].sample() for i in env.agents}
            _, _, _, _, done, _ = env.step(action)

        if len(frames) > STEPS:
            break

    env.close()

    frame = frames[-1]
    if resize:
        # h / w = H / w'
        # w' = Hw/h
        frame = frame.resize((HEIGHT, int(HEIGHT * frame.width / frame.height)))

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png
    frame.save(
        v_file_path,
        dpi=(120, 120),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="Name of environments to run.",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Whether to resize images to standard size.",
    )
    args = parser.parse_args()
    if args.full_env_ids == ["all"]:
        args.full_env_ids = [
            "CooperativeReaching-v0",
            "Driving-v1",
            "LevelBasedForaging-v3",
            "PredatorPrey-v0",
            "PursuitEvasion-v1_i0",
            "PursuitEvasion-v1_i1",
        ]

    for full_env_id in args.full_env_ids:
        gen_img(full_env_id, args.resize)
        print(
            f"Saved image for {full_env_id} to "
            f"{exp_utils.RESULTS_DIR / (full_env_id + '.pdf')}"
        )
