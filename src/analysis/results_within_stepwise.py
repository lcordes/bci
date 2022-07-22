import json
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


def plot_within_stepwise(
    train_accs, val_accs, base_train_accs, base_val_accs, within_accs, train_size, clf
):
    plt.close()
    plt.figure(figsize=(16, 7))
    x = range(1, len(train_accs) + 1)
    for y, label in zip(
        [train_accs, val_accs, base_train_accs, base_val_accs, within_accs],
        ["train", "val", "base_train", "base_val", "within"],
    ):
        plt.plot(x, y, label=label)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.grid()
    plt.title(f"Within stepwise selection for {clf} with train_size={train_size}")
    plt.legend()
    plt.savefig(
        f"{RESULTS_PATH}/within_stepwise/Within_stepwise_{clf}_{train_size}.png",
        dpi=400,
    )


def within_stepwise_results():
    dir = Path(f"{RESULTS_PATH}/within_stepwise")
    files = [path for path in dir.glob("*within*.json")]
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        print(f"{data['classifier']} with train_size={data['train_size']}")
        train_accs = [u["train_acc"] for u in data["users"]]
        val_accs = [u["val_acc"] for u in data["users"]]
        base_train_accs = [u["base_train_acc"] for u in data["users"]]
        base_val_accs = [u["base_val_acc"] for u in data["users"]]
        within_accs = [u["within_acc"] for u in data["users"]]
        plot_within_stepwise(
            train_accs,
            val_accs,
            base_train_accs,
            base_val_accs,
            within_accs,
            data["train_size"],
            data["classifier"],
        )
        print(f"Training accuracy: {np.mean(train_accs):.2f}")
        print(f"Validation accuracy: {np.mean(val_accs):.2f}")
        print(f"Baseline training accuracy: {np.mean(base_train_accs):.2f}")
        print(f"Baseline validation accuracy: {np.mean(base_val_accs):.2f}")
        print(f"Within accuracy: {np.mean(within_accs):.2f}\n")


if __name__ == "__main__":

    within_stepwise_results()
