import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np

from natsort import natsorted
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

def plot_overall_results(results, users, y_labels, title):
    x_size = 15 if results.shape[1] > 10 else 9
    plt.figure(figsize=(x_size, 3))
    heatmap(
        results,
        vmin=np.min(results) - 0.15,
        vmax=np.max(results) + 0.15,
        annot=True,
        xticklabels=users + ["avg"],
        yticklabels=y_labels
    )
    plt.title(title)
    plt.savefig(
        f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
    )



if __name__ == "__main__":

        # Training
        dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
        users = natsorted([path.stem for path in dir.glob("*.hdf5")])
        results = np.zeros((4, len(users)+1))
        n_users = len(users)
        with open(f"{RESULTS_PATH}/transfer_learning/Between classification (training data, 10-12).npy", 'rb') as f:
            results[:2,:] = np.load(f)
        with open(f"{RESULTS_PATH}/within/Within classification 50-50 split (training data, 10-12, 100 reps).npy", 'rb') as f:
            results[2, :n_users] = np.mean(np.load(f), axis=1)
            results[2, n_users] = np.mean(results[2, :n_users])
        with open(f"{RESULTS_PATH}/within/Within classification loocv(training data, 10-12).npy", 'rb') as f:
            results[3, :n_users] = np.load(f)
            results[3, n_users] = np.mean(results[3, :n_users])

        plot_overall_results(results, users, 
        ["Between Base", "Between TF", "Within 50-50", "Within LOOCV"],
        f"Overall results (training data, 10-12)"
        )

        #Benchmark
        dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
        users = natsorted([path.stem for path in dir.glob("*.gdf") if path.stem[-1] == "T"])
        
        results = np.zeros((4, len(users)+1))
        n_users = len(users)
        with open(f"{RESULTS_PATH}/transfer_learning/Between classification (benchmark data, 8-30).npy", 'rb') as f:
            results[:2,:] = np.load(f)
        with open(f"{RESULTS_PATH}/within/Within classification 50-50 split (benchmark data, 8-30, 100 reps).npy", 'rb') as f:
            results[2, :n_users] = np.mean(np.load(f), axis=1)
            results[2, n_users] = np.mean(results[2, :n_users])
        with open(f"{RESULTS_PATH}/within/Within classification loocv(benchmark data, 8-30).npy", 'rb') as f:
            results[3, :n_users] = np.load(f)
            results[3, n_users] = np.mean(results[3, :n_users])

        plot_overall_results(results, users, 
        ["Between Base", "Between TF", "Within 50-50", "Within LOOCV"],
        f"Overall results (benchmark data, 8-30)"
        )

