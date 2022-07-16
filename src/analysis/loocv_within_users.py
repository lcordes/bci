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


def score_loocv(recording_name, constructor, config):
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
    classes = config["n_classes"]
    conf_matrices = [np.zeros((classes, classes)) for _ in range(3)]

    for row, constructor in enumerate(constructors):
        for column, recording_name in enumerate(users):
            metrics = score_loocv(recording_name, constructor, config)
            data[row, column] = metrics["acc"]
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


def save_loocv_plot(users, classifiers, config, title):
    fig, _ = plt.subplots(1, 1, figsize=(15, 3))
    axes = fig.axes

    loocv_plot(users, classifiers, axes[0], config)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(f"LOOCV {title}.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/users")
    users = sorted([path.stem for path in dir.glob("*.hdf5")])

    classifiers = [LDAClassifier, SVMClassifier]
    config = create_config()
    title = "Accuracy_LDA_SVM, 2_comps"  # TODO autogenerate title based on non-default config params

    save_loocv_plot(users, classifiers, config, title)
