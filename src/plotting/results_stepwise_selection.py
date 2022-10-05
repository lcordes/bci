import json
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


def plot_stepwise_history(data, mean_acc):
    plt.close()
    plt.figure(figsize=(13, 7))
    x = range(1, len(data["users"]))
    for i, u in enumerate(data["users"]):
        y = u["accs"]
        plt.plot(x, y, label=u["user"])
        plt.plot(len(u["best_set"]), u["best_acc"], "x", c="red", markersize=10)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.grid()
    plt.title(
        f"{data['classifier']} stepwise selection of optimal train users (optimal mean={mean_acc})"
    )
    plt.savefig(
        f"{RESULTS_PATH}/stepwise_selection/Acc_trajectory_{data['classifier']}.png",
        dpi=400,
    )


def stepwise_results(clf_name):
    """Get summary statistics of prediction accuracies across test users,
    how often each user was included in an optimal train set and plot the
    stepwise accuracy development."""

    with open(f"{RESULTS_PATH}/stepwise_selection/{clf_name}_optimal.json", "r") as f:
        data = json.load(f)

    mean_acc = np.round(np.mean([user["best_acc"] for user in data["users"]]), 3)
    std = np.round(np.std([user["best_acc"] for user in data["users"]]), 3)

    for user in data["users"]:
        user["n_selected"] = sum(
            [user["user"] in other_user["best_set"] for other_user in data["users"]]
        )

    plot_stepwise_history(data, mean_acc)
    return data


def stepwise_compare_clfs(clf1_name, clf2_name):
    with open(f"{RESULTS_PATH}/stepwise_selection/{clf1_name}_optimal.json", "r") as f:
        clf1_data = json.load(f)
    with open(f"{RESULTS_PATH}/stepwise_selection/{clf2_name}_optimal.json", "r") as f:
        clf2_data = json.load(f)

    agreement = [
        set(u_clf1["best_set"]).intersection(set(u_clf2["best_set"]))
        for u_clf1, u_clf2 in zip(clf1_data["users"], clf2_data["users"])
    ]
    total = []
    for i, a in enumerate(agreement):
        total.extend(list(a))

    n_selected = [total.count(f"u{i}") for i in range(1, 21)]
    for i in np.flip(np.argsort(n_selected)):
        clf1_num = clf1_data["users"][i]["n_selected"]
        clf2_num = clf2_data["users"][i]["n_selected"]
        print(
            f"u{i+1}: {n_selected[i]} agreed inclusions ({clf1_data['classifier']}={clf1_num}, {clf2_data['classifier']}={clf2_num})"
        )


if __name__ == "__main__":

    stepwise_results("LDA")
    # stepwise_compare_clfs("LDA", "SVM")
