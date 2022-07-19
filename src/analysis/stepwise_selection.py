import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from classification.classifiers import (
    LDAClassifier,
    SVMClassifier,
    RFClassifier,
    MLPClassifier,
)
from classification.train_test_model import train_model, test_model, create_config
import numpy as np
from natsort import natsorted
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def get_current_best_user(users, test_user, classifier, config, selected):
    """Return the user which best predicts the test_user, as well as their
    prediction accuracy. If selected is not empty the best new user is
    determined in combination with the already selected users."""

    accs = np.zeros(len((users)))
    for idx, train_user in enumerate(users):
        if train_user != test_user and not train_user in selected:
            if selected:
                train_user = [train_user] + selected
            model = train_model(train_user, classifier, config)
            accs[idx] = test_model(test_user, model)
    max_idx = np.argmax(accs)
    return users[max_idx], accs[max_idx]


def stepwise_selection(users, test_user, classifier, config):
    """Sequentially add the current most predictive user to the
    train set and save the resulting prediction accuracies."""
    selected = []
    accs = []
    step = 1

    print(f"Finding optimal train set users for testing on {test_user}")
    while step < len(users):
        best_user, acc = get_current_best_user(
            users, test_user, classifier, config, selected
        )
        selected.append(best_user)
        accs.append(acc)
        print(f"Step {step}: Adding {best_user}: acc = {np.round(acc, 3)}")
        step += 1
    return accs, selected


def get_all_optimal_train_sets(users, classifier, config, save_model=False):
    """Determine the best combination of users to train on for all users individually"""
    data = {"users": [], "classifier": (classifier.__name__).replace("Classifier", "")}
    for user in users:
        config["model_name"] = f"optimal_{user}"
        accs, selected = stepwise_selection(users, user, classifier, config)
        best_set = selected[: (np.argmax(accs) + 1)]
        print("Accuracy history:", [np.round(a, 3) for a in accs])
        print(f"Final model for {user} selected: {best_set}")

        model = train_model(best_set, classifier, config, save=save_model)
        best_acc = test_model(user, model)
        print("Final acc:", np.round(best_acc, 3), "\n")
        data["users"].append(
            {
                "user": user,
                "best_acc": best_acc,
                "best_set": best_set,
                "accs": accs,
                "selected": selected,
            }
        )

    clf_name = str(classifier.__name__).replace("Classifier", "")
    with open(f"{RESULTS_PATH}/stepwise_selection/{clf_name}_optimal.json", "w") as f:
        json.dump(data, f)
    return data


def optimal_train_sets_info(data):
    """Report summary statistics of prediction accuracies across test users,
    how often each user was included in an optimal train set and plot the
    stepwise accuracy development."""

    if isinstance(data, str):
        with open(f"{RESULTS_PATH}/stepwise_selection/{data}_optimal.json", "r") as f:
            data = json.load(f)

    print(f"Summary for {data['classifier']}")
    mean_acc = np.round(np.mean([user["best_acc"] for user in data["users"]]), 3)
    std = np.round(np.std([user["best_acc"] for user in data["users"]]), 3)
    print(f"Mean accuracy: {mean_acc}, std: {std}")
    selected = []
    for user in data["users"]:
        selected.extend(user["best_set"])
    n_selected = [selected.count(u["user"]) for u in data["users"]]

    for i in np.flip(np.argsort(n_selected)):
        print(f"u{i+1}: {n_selected[i]} inclusions")
    plot_acc_history(data, mean_acc)


def plot_acc_history(data, mean_acc):
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


def get_results(classifiers, config, rerun=True):
    if rerun:
        results = [
            get_all_optimal_train_sets(users, classifier, config, save_model=True)
            for classifier in classifiers
        ]
    else:
        results = [
            classifier.__name__.replace("Classifier", "") for classifier in classifiers
        ]

    for result in results:
        optimal_train_sets_info(result)


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/users")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    params = {}
    config = create_config(**params)
    classifiers = [LDAClassifier, SVMClassifier]

    get_results(classifiers, config, rerun=False)
