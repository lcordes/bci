import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
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


def compile_results(users, between, within_half, within_loocv):
    n_users = len(users)
    results = np.zeros((4, n_users+1))
    with open(f"{RESULTS_PATH}/transfer_learning/{between}.npy", 'rb') as f:
        results[:2,:] = np.load(f)
    with open(f"{RESULTS_PATH}/within/{within_half}.npy", 'rb') as f:
        results[2, :n_users] = np.mean(np.load(f), axis=1)
        results[2, n_users] = np.mean(results[2, :n_users])
    with open(f"{RESULTS_PATH}/within/{within_loocv}.npy", 'rb') as f:
        results[3, :n_users] = np.load(f)
        results[3, n_users] = np.mean(results[3, :n_users])
    return results


def plot_overall_results(results, users, config, save=False):
    x_size = 15 if results.shape[1] > 10 else 9
    plt.figure(figsize=(x_size, 3))
    heatmap(
        results,
        vmin=np.min(results) - 0.15,
        vmax=np.max(results) + 0.15,
        annot=True,
        xticklabels=users + ["avg"],
        yticklabels=["Between Base", "Between TF", "Within 50-50", "Within LOOCV"]
    )
    title = f"Overall results ({config['data_set']} data)"
    plt.title(title)

    if save:
        plt.savefig(
            f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
        )
    else:
        plt.show()


if __name__ == "__main__":

        # Training
        config = create_config({"data_set": "training"})
        users = get_users(config)

        between = "Between classification (training data, 10-12)"
        within_half = "Within classification 50-50 split (training data, 10-12, 100 reps)"
        within_loocv = "Within classification loocv(training data, 10-12)"
        results = compile_results(users, between, within_half, within_loocv)
        plot_overall_results(results, users, config)

        #Benchmark
        config = create_config({"data_set": "benchmark"})
        users = get_users(config)

        between = "Between classification (benchmark data, 8-30)"
        within_half = "Within classification 50-50 split (benchmark data, 8-30, 100 reps)"
        within_loocv = "Within classification loocv(benchmark data, 8-30)"
        results = compile_results(users, between, within_half, within_loocv)
        plot_overall_results(results, users, config)

