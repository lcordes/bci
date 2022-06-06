import argparse
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition.preprocessing import preprocess_recording
from classifiers import LDAClassifier, SVMClassifier, RFClassifier
from train_model import train_model
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import axis as ax

import mne

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def score_loocv(constructor, X, y):
    looc = LeaveOneOut()
    looc.get_n_splits(X)
    scores = []
    for train_index, test_index in looc.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = constructor()
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return np.round(np.mean(scores), 3)


def loocv_per_block(constructor, X, y):
    overall_acc = score_loocv(constructor, X, y)
    print(f"Overall loocv mean accuracy: {overall_acc}")

    for block in range(1, 4):
        block_len = len(y) // 3
        start = (block - 1) * block_len
        stop = block * block_len
        block_acc = score_loocv(constructor, X[start:stop, :], y[start:stop])
        print(f"Block {block} loocv mean accuracy: {block_acc}")


def test_model(recording_name, model_name, type):
    extractor = CSPExtractor()
    extractor.load_model(model_name)
    predictor = (
        SVMClassifier()
        if type == "SVM"
        else RFClassifier()
        if type == "RF"
        else LDAClassifier()
    )
    predictor.load_model(model_name)
    model_info = predictor.model.model_info

    X, y = preprocess_recording(recording_name, model_info)
    X_transformed = extractor.fit_transform(X, y)
    acc = predictor.score(X_transformed, y)
    return np.round(acc, 3)


def test_heatmap(participants, type="RF"):
    data = np.zeros((len(participants), len(participants)))
    for row, model_name in enumerate(participants):
        train_model(model_name, type=type)
        for column, recording_name in enumerate(participants):
            data[row, column] = test_model(recording_name, model_name, type)

    fig, ax = plt.subplots(1, 1)

    # Show the tick labels
    ax.xaxis.set_tick_params(labeltop=True)

    # Hide the tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    map = sns.heatmap(
        data,
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=participants,
        yticklabels=participants,
    )
    plt.xlabel("Recordings")
    plt.ylabel("Models")
    return map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", help="Name of the model to be tested.")

    parser.add_argument("recording_name", help="Name of the recording to test on.")

    args = parser.parse_args()
    recording_name = args.recording_name.replace(".hdf5", "")
    model_name = args.model_name.replace(".pkl", "")

    no_inputs = (
        True  # if no recroding and model give, create heatmap of all available models
    )

    if no_inputs:
        participants = [
            "lars",
            "larstap",
            "pilot1",
            "pilot2",
        ]  # get all models in directory instead
        heatmap = test_heatmap(participants, type="SVM")
        plt.savefig("heatmap.png")

    else:
        test_model(recording_name, model_name, type="LDA")
