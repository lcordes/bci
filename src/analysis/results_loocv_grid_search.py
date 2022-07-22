import json
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from seaborn import heatmap

load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


def loocv_grid_search_results(n_best=10):
    dir = Path(f"{RESULTS_PATH}/loocv_grid_search")
    files = [path for path in dir.glob("*.json")]

    configs = []
    for file in files:
        with open(file, "r") as f:
            configs.extend(json.load(f))

        descriptions, accs = [], []
        for c in configs:
            user_accs = [
                accuracy_score(user["y_true"], user["y_pred"]) for user in c["users"]
            ]
            accs.append(user_accs)
            descriptions.append(c["config"]["description"])

    mean_accs = [np.mean(u_accs) for u_accs in accs]
    best_accs_idx = list(np.flip(np.argsort(mean_accs)))[:n_best]
    for i in best_accs_idx:
        print(f"Accuracy: {np.round(mean_accs[i], 3)}")
        print(descriptions[i], "\n")

    best_accs = [accs[i] for i in best_accs_idx]
    labels = [f"Config {i+1}" for i in best_accs_idx]
    plot_grid_search_results(best_accs, labels)


def plot_grid_search_results(user_accs, y_labels):

    data = np.stack([np.array(config) for config in user_accs])

    # Add row (user) means as column
    data = np.concatenate((np.mean(data, axis=1).reshape(-1, 1), data), axis=1)
    x_labels = ["avg"] + [f"u{i+1}" for i in range(data.shape[1] - 1)]

    plt.figure(figsize=(15, len(user_accs) // 2))
    heatmap(
        data,
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
    )
    title = "Grid_search_results"
    plt.title(title)
    plt.savefig(
        f"{RESULTS_PATH}/loocv_grid_search/{title}.png", dpi=400, bbox_inches="tight"
    )


if __name__ == "__main__":

    loocv_grid_search_results()
