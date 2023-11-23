import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    results = pd.read_csv(args.results_file)

    sns.set_theme()
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")
    sns.catplot(
        data=results,
        x="train_pop",
        y="mean_returns",
        hue="eval_pop",
        kind="box",
        height=4,
        aspect=1.5,
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="BR results .csv file.",
    )
    main(parser.parse_args())
