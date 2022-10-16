import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.utilities import create_config
from pipeline.preprocessing import get_users
from analysis.between_users_transfer import between_classification
from analysis.within_loocv import within_loocv
from analysis.within_train_test import within_train_test


def compile_results(config, title, save=False):
    users = get_users(config)
    if config ["data_set"] == "evaluation":
        users = [f"E{i+1}" for i in range(len(users))]
    df = pd.DataFrame(columns=users)

    print("Working on between users transfer")
    transfer_data_set = "training" if config["data_set"] == "evaluation" else "within"
    df.loc["Between Base"], df.loc["Between TF"] = between_classification(config, transfer_data_set=transfer_data_set)
    print("\nWorking on within 50-50")
    df.loc["Within 50-50"] = within_train_test(config, train_size=0.5)
    print("\nWorking on within 67-33")
    df.loc["Within 67-33"] = within_train_test(config, train_size=0.67)
    print("\nWorking on within loocv")
    df.loc["Within LOOCV"] = within_loocv(config)
    df["avg"] = df.mean(axis=1)

    if save:
        df.to_csv(f"{RESULTS_PATH}/overall/{title}.csv", index_label="Data Split")  
    return df


def plot_overall_results(df, title, save=False):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(f"{RESULTS_PATH}/overall/{title}.csv", index_col=0)
    x_size = 15 if df.shape[1] > 10 else 9
    plt.figure(figsize=(x_size, 3))
    heatmap(
        df,
        vmin=df.max().max() + 0.15,
        vmax=df.min().min() - 0.15,
        annot=True,
    )
    plt.title(title)
    plt.ylabel(None)
    if save:
        plt.savefig(f"{RESULTS_PATH}/overall/{title}.png", dpi=400, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    for data_set in ["training", "evaluation", "benchmark"]:    
        config = create_config({"data_set": data_set})
        title = f"Overall results ({data_set} data)"
        df = compile_results(config, title, save=True)
        plot_overall_results(df, title, save=True)