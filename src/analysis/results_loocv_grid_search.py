import argparse
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from seaborn import heatmap

load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


def load_grid_search_results(most_recent=False):
    dir = Path(f"{RESULTS_PATH}/loocv_grid_search")
    files = sorted([path for path in dir.glob("*.json")], key=os.path.getmtime)
    if most_recent:
        files = [files[-1]]

    configs = []
    for file in files:
        with open(file, "r") as f:
            c = json.load(f)
            if c and len(c[0]["users"]) == 20:
                configs.extend(c)
    params, accs = [], []
    for c in configs:
        user_accs = [
            accuracy_score(user["y_true"], user["y_pred"]) for user in c["users"]
        ]
        accs.append(user_accs)
        params.append(c["config"])

    all_accs = np.stack(accs)
    print(
        "Current optimal model:",
        params[np.argmax(np.mean(all_accs, axis=1))]["description"],
    )

    return all_accs, params


def violin_plot(data):

    opt_clf = np.argmax(np.mean(data, axis=1))
    _, ax = plt.subplots(figsize=(10, 6))
    vp = ax.violinplot(data, showmeans=True)
    line = ax.plot(range(1, 21), data[opt_clf, :], label="Optimal overall classifier")

    plt.gca().xaxis.set_major_locator(FixedLocator(range(1, 21)))
    plt.grid()
    plt.title(f"Density of different classifier accuracies per user")
    plt.xlabel("Users")
    plt.ylabel("Accuracy")
    plt.legend(handles=line, loc="upper left")
    plt.savefig(
        f"{RESULTS_PATH}/loocv_grid_search/Clf_results_across_users.png",
        dpi=400,
    )


# TODO update functions below


def grid_search_results(data, configs, n_best=10):
    print("Range of accuracies:")
    user_best = data.max(axis=0)
    for i in np.argmax(data, axis=0):
        print(configs[i]["description"], "\n")
    mean_accs = [np.mean(u_accs) for u_accs in data]
    best_accs_idx = list(np.flip(np.argsort(mean_accs)))[:n_best]
    # for i in best_accs_idx:
    #     print(f"Accuracy: {np.round(mean_accs[i], 3)}")
    #     print(descriptions[i], "\n")

    best_accs = [data[i] for i in best_accs_idx]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recent",
        help="Only show results from the most recent run",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # loocv_grid_search_results(most_recent=args.recent)
    data, configs = load_grid_search_results()
    violin_plot(data)
