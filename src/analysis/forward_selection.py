import sys
from pathlib import Path
import json


parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition.preprocessing import preprocess_recording
from data_acquisition.rail_check import railed_heatmap
from classifiers import LDAClassifier, SVMClassifier, RFClassifier, MLPClassifier
from train_model import train_model, create_config
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import seaborn as sns
from matplotlib import pyplot as plt
import mne

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def get_best_train_user(users, test_user, classifier, config, included):
    accs = np.zeros(len((users)))
    for idx, train_user in enumerate(users):
        if train_user != test_user and not train_user in included:
            # print(f"Testing user {idx + 1} for inclusion")

            if included:
                train_user = [train_user] + included
            train_model(train_user, classifier, config)
            accs[idx] = test_model(test_user, config["model_name"], classifier)
    return users[np.argmax(accs)], accs[np.argmax(accs)]


def get_optimal_train_users(users, test_user, classifier, config):
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
                f"Step {step}: Did not add user {users.index(best_user) + 1}: acc = {acc}"
            )

            break
    indices = [users.index(i) + 1 for i in included]
    print(f"Model for user {users.index(test_user) + 1} included: {indices}")
    return included


def forward_selection(users, classifier, config):
    total_included = []
    for i, test_user in enumerate(users):
        included = get_optimal_train_users(users, test_user, classifier, config)
        total_included.extend(included)
    counts = [total_included.count(p) for p in users]
    indices = list(np.argsort(counts))
    for i in indices:
        print(f"user {users[i+1]} in {counts[i]} models")


def get_optimal(clf):
    data = {"users": [], "classifier": clf.__name__}
    for user in users:
        config = create_config(model_name="tmp")
        optimal = get_optimal_train_users(users, user, clf, config)
        train_model(optimal, clf, config)
        acc = test_model(user, config["model_name"], clf)
        print("Final acc:", config["model_name"], acc)
        new_user = {"user": user, "acc": acc, "optimal": optimal}
        data["users"].append(new_user)

    with open(f"{clf.__name__}_optimal.json", "w") as f:
        json.dump(data, f)


def check_optimal(file):
    with open(file) as f:
        data = json.load(f)
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
    users = sorted([path.stem for path in dir.glob("*.hdf5")])

    classifiers = [LDAClassifier, SVMClassifier, RFClassifier]

    config = create_config()
    title = "Accuracy_LDA_SVM"
    # TODO autogenerate title based on non-default config params

    forward_selection(users, SVMClassifier, config)

    check_optimal("LDAClassifier_optimal.json")
    get_optimal(LDAClassifier)
