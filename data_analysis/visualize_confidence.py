import argparse
from os.path import basename

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answer_files", nargs="+")
    args = parser.parse_args()

    dfs = {}
    for x in args.answer_files:
        name = basename(x)
        name = name[:name.rfind(".")]
        dfs[name] = pd.read_csv(x)

    for k, df in dfs.items():
        df = df[df["n_answers"] > 0]
        plt.hist(df["predicted_score"] - df["predicted_score"].mean(), 50, label=k, alpha=0.5)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
