import sys
from pathlib import Path
import json

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from pipeline.utilities import train_model, test_model, create_config
import numpy as np
from natsort import natsorted
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def get_current_best_user(users, test_user, config, selected, subset_idx=None):
    """Return the user which best predicts the test_user, as well as their
    prediction accuracy. If selected is not empty the best new user is
    determined in combination with the already selected users."""

    accs = np.zeros(len((users)))
    for idx, train_user in enumerate(users):
        if train_user != test_user and not train_user in selected:
            if selected:
                train_user = [train_user] + selected
            model = train_model(train_user, config)
            accs[idx] = test_model(test_user, model, subset_idx)
    max_idx = np.argmax(accs)
    return users[max_idx], accs[max_idx]


def stepwise_selection(users, test_user, config, subset_idx=None):
    """Sequentially add the current most predictive user to the
    train set and save the resulting prediction accuracies."""
    selected = []
    accs = []
    step = 1

    print(f"Finding optimal train set users for testing on {test_user}")
    while step < len(users):
        best_user, acc = get_current_best_user(
            users, test_user, config, selected, subset_idx
        )
        selected.append(best_user)
        accs.append(acc)
        print(f"Step {step}: Adding {best_user}: acc = {np.round(acc, 3)}")
        step += 1
    return accs, selected


def get_all_optimal_train_sets(users, config, save_model=False):
    """Determine the best combination of users to train on for all users individually"""
    data = {"users": [], "classifier": config["model_type"]}
    for user in users:
        config["model_name"] = f"{config['model_type']}_optimal_{user}"
        accs, selected = stepwise_selection(users, user, config)
        best_set = selected[: (np.argmax(accs) + 1)]
        print("Accuracy history:", [np.round(a, 3) for a in accs])
        print(f"Final model for {user} selected: {best_set}")

        model = train_model(best_set, config, save=save_model)
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

    with open(
        f"{RESULTS_PATH}/stepwise_selection/{config['model_type']}_optimal.json", "w"
    ) as f:
        json.dump(data, f)
    return data


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    params = {
        "model_type": "LDA",
        "clf_specific": {"solver": "lsqr", "shrinkage": "auto"},
    }
    config = create_config(**params)
    get_all_optimal_train_sets(users, config, save_model=True)
