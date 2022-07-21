import json
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


def loocv_grid_search_results(n_best=None):
    dir = Path(f"{RESULTS_PATH}/loocv_within_users")
    files = [path for path in dir.glob("*.json")]

    configs = []
    for file in files:
        with open(file, "r") as f:
            configs.extend(json.load(f))

        descriptions, accs = [], []
        for c in configs:
            user_accs = [
                accuracy_score(user["y_true"], user["y_pred"])
                for user in c["users"]
                if user["user"] == "u17"
            ]
            accs.append(np.mean(user_accs))
            descriptions.append(c["config"]["description"])

    best_accs_idx = np.flip(np.argsort(accs))
    if n_best:
        best_accs_idx = best_accs_idx[:10]
    for i in best_accs_idx:
        print(f"Accuracy: {np.round(accs[i], 3)}")
        print(descriptions[i], "\n")


def plot_within_stepwise(train_accs, val_accs, within_accs, train_size, clf):
    plt.close()
    plt.figure(figsize=(16, 7))
    x = range(1, len(train_accs) + 1)
    for y, label in zip(
        [train_accs, val_accs, within_accs], ["train", "val", "within"]
    ):
        plt.plot(x, y, label=label)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.grid()
    plt.title(f"Within stepwise selection for {clf} with train_size={train_size})")
    plt.legend()
    plt.savefig(
        f"{RESULTS_PATH}/stepwise_selection/Within_stepwise_{clf}_{train_size}.png",
        dpi=400,
    )


def within_stepwise_results():
    dir = Path(f"{RESULTS_PATH}/stepwise_selection")
    files = [path for path in dir.glob("*within*.json")]
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        print(f"{data['classifier']} with train_size={data['train_size']}")
        train_accs = [u["train_acc"] for u in data["users"]]
        val_accs = [u["val_acc"] for u in data["users"]]
        within_accs = [u["within_acc"] for u in data["users"]]
        plot_within_stepwise(
            train_accs, val_accs, within_accs, data["classifier"], data["train_size"]
        )
        print(f"Training accuracy: {np.mean(train_accs):.2f}")
        print(f"Validation accuracy: {np.mean(val_accs):.2f}")
        print(f"Within accuracy: {np.mean(within_accs):.2f}\n")


if __name__ == "__main__":
    # loocv_grid_search_results(n_best=10)
    within_stepwise_results()
