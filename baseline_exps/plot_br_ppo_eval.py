import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    results = pd.read_csv(args.results_file)

    results.rename(
        columns={
            "train_pop": "Train Population",
            "eval_pop": "Test Population",
            "mean_returns": "Mean Return",
        },
        inplace=True,
    )

    sns.set_theme()
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")
    plot = sns.catplot(
        data=results,
        x="Train Population",
        y="Mean Return",
        hue="Test Population",
        kind="box",
        height=4,
        aspect=1.5,
    )

    if args.save_path is not None:
        plot.figure.savefig(args.save_path, bbox_inches="tight")

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
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save plot too.",
    )
    main(parser.parse_args())
