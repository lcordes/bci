import sys
from pathlib import Path
import json


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
import mne

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def get_best_train_user(users, test_user, classifier, config, included):
    accs = np.zeros(len((users)))
    for idx, train_user in enumerate(users):
        if train_user != test_user and not train_user in included:
            if included:
                train_user = [train_user] + included
            model = train_model(train_user, classifier, config)
            accs[idx] = test_model(test_user, model)
    return users[np.argmax(accs)], accs[np.argmax(accs)]


def get_best_train_set(users, test_user, classifier, config):
    best_acc = 0
    included = []
    step = 0
    print(
        f"Finding optimal train set users for testing on user {users.index(test_user) + 1}"
    )
    while True:
        best_user, acc = get_best_train_user(
            users, test_user, classifier, config, included
        )
        step += 1

        if acc >= best_acc:
            best_acc = acc
            included.append(best_user)
            print(f"Step {step}: Adding user {users.index(best_user) + 1}: acc = {acc}")
        else:
            print(
                f"Step {step}: Did not add user {users.index(best_user) + 1}: acc = {acc}\n"
            )

            break
    indices = [users.index(i) + 1 for i in included]
    print(f"Model for user {users.index(test_user) + 1} included: {indices}")
    return included


def get_all_optimal_train_sets(users, classifier, save_models=False):
    data = {"users": [], "classifier": classifier.__name__}
    for user in users:
        config = create_config(model_name=f"optimal_{user}")
        optimal = get_best_train_set(users, user, classifier, config)
        model = train_model(optimal, classifier, config, save=save_models)
        acc = test_model(user, model)
        print("Final acc:", acc, "\n")
        new_user = {"user": user, "acc": acc, "optimal": optimal}
        data["users"].append(new_user)
    clf_name = str(classifier.__name__).replace("Classifier", "")
    optimal_train_sets_info(data)

    with open(f"{RESULTS_PATH}/forward_selection/{clf_name}_optimal.json", "w") as f:
        json.dump(data, f)


def optimal_train_sets_info(data):
    print("Summary")
    acc = np.round(np.mean([user["acc"] for user in data["users"]]), 3)
    std = np.round(np.std([user["acc"] for user in data["users"]]), 3)
    print(f"Mean accuracy: {acc}, std: {std}")
    included = []
    for user in data["users"]:
        included.extend(user["optimal"])
    n_included = [included.count(u["user"]) for u in data["users"]]

    for i in np.flip(np.argsort(n_included)):
        print(f"User {i+1}: {n_included[i]} inclusions")


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/users")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    params = {}
    config = create_config(**params)

    get_all_optimal_train_sets(users, LDAClassifier, save_models=True)
