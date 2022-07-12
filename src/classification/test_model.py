import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition.preprocessing import preprocess_recording
from data_acquisition.rail_check import railed_heatmap
from classifiers import LDAClassifier, SVMClassifier, RFClassifier
from train_model import train_model, create_config
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from matplotlib import pyplot as plt
import mne

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def test_model(test_recording, model_name, model_constructor):
    if isinstance(model_name, list):
        model_name = "merged_model"
    extractor = CSPExtractor()
    extractor.load_model(model_name)
    predictor = model_constructor()
    predictor.load_model(model_name)
    config = predictor.model.config

    X, y = preprocess_recording(test_recording, config)
    X_transformed = extractor.fit_transform(X, y)
    acc = predictor.score(X_transformed, y)
    return np.round(acc, 3)


def test_heatmap(users, constructor, ax):
    data = np.zeros((len(users) + 1, len(users)))
    for row, model_name in enumerate(users):
        train_model(model_name, constructor)
        for column, recording_name in enumerate(users):
            data[row, column] = test_model(recording_name, model_name, constructor)

    for column, recording_name in enumerate(users):
        data[len(users), column] = score_loocv(recording_name, constructor)["acc"]

    # Show the tick labels
    ax.xaxis.set_tick_params(labeltop=True)

    labels = [p.replace("Training_session_", "")[:7] for p in users]
    # Hide the tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    map = sns.heatmap(
        data,
        ax=ax,
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=labels,
        yticklabels=labels + ["LOOCV"],
    )
    ax.set_title(f"{constructor.__name__} Accuracy")
    # plt.xlabel("Recordings")
    # plt.ylabel("Models")


def score_loocv(recording_name, constructor, config, conf_mat=False):
    extractor = CSPExtractor()
    extractor.load_model(recording_name)
    X, y = preprocess_recording(recording_name, config)
    X_transformed = extractor.fit_transform(X, y)

    looc = LeaveOneOut()
    looc.get_n_splits(X_transformed)
    y_true = []
    y_pred = []
    scores = []

    for train_index, test_index in looc.split(X_transformed):
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = constructor()
        clf.fit(X_train, y_train)
        y_true.append(y_test)
        y_pred.append(clf.predict(X_test))
        scores.append(clf.score(X_test, y_test))
    from sklearn.metrics import matthews_corrcoef

    return {
        "acc": np.round(np.mean(scores), 3),
        "conf": confusion_matrix(y_true, y_pred),
        "matthews": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def loocv_per_block(constructor, X, y):
    overall_acc = score_loocv(constructor, X, y)["acc"]
    print(f"Overall loocv mean accuracy: {overall_acc}")

    for block in range(1, 4):
        block_len = len(y) // 3
        start = (block - 1) * block_len
        stop = block * block_len
        block_acc = score_loocv(constructor, X[start:stop, :], y[start:stop])["acc"]
        print(f"Block {block} loocv mean accuracy: {block_acc}")


def loocv_plot(users, constructors, ax, config):
    data = np.zeros((len(classifiers), len(users)))
    conf_matrices = [np.zeros((3, 3)) for _ in range(3)]

    for row, constructor in enumerate(constructors):
        for column, recording_name in enumerate(users):
            metrics = score_loocv(recording_name, constructor, config)
            data[row, column] = metrics["f1"]
            conf_matrices[row] = conf_matrices[row] + metrics["conf"]
            print(f"Classifier {row+1} user {column+1} done")

    # Add row and column means
    data = np.concatenate((data, np.mean(data, axis=0).reshape(1, -1)), axis=0)
    data = np.concatenate((data, np.mean(data, axis=1).reshape(-1, 1)), axis=1)

    # Show the tick labels
    ax.xaxis.set_tick_params(labeltop=True)

    x_labels = [f"P{i}" for i in range(1, 21)] + ["avg"]
    y_labels = ["LDA", "SVM", "RF"][: len(constructors)] + ["avg"]
    # Hide the tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    sns.heatmap(
        data,
        ax=ax,
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
    )


def save_heatmap_plot(users, classifiers, config, title):
    fig, _ = plt.subplots(2, 2, figsize=(24, 20))
    axes = fig.axes

    for i, classifier in enumerate(classifiers):
        test_heatmap(users, classifier, axes[i])
        print(f"{classifier.__name__} results done.")

    railed_heatmap(users, axes[-1])
    fig.tight_layout()
    plt.title(title)
    plt.savefig(f"Heatmap {title}.png", dpi=400)


def save_loocv_plot(users, classifiers, config, title):
    fig, _ = plt.subplots(1, 1, figsize=(15, 3))
    axes = fig.axes

    loocv_plot(users, classifiers, axes[0], config)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(f"LOOCV {title}.png", dpi=400, bbox_inches="tight")


def get_best_train_user(users, test_user, classifier, config, included):
    accs = np.zeros(len((users)))
    for idx, train_user in enumerate(users):
        if train_user != test_user and not train_user in included:
            # print(f"Testing user {idx + 1} for inclusion")

            if included:
                train_user = [train_user] + included
            train_model(train_user, classifier, config)
            accs[idx] = test_model(test_user, train_user, classifier)
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


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/participants")
    users = sorted([path.stem for path in dir.glob("*.hdf5")])

    # classifiers = [LDAClassifier, SVMClassifier, RFClassifier]
    classifiers = [LDAClassifier, SVMClassifier]

    config = create_config(
        imagery_window=4,
    )
    title = "F1 score, imagery_window=4"

    # save_heatmap_plot(users, classifiers, config, title)
    # save_loocv_plot(users, classifiers, config, title)
    forward_selection(users, SVMClassifier, config)
