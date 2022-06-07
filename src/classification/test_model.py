import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition.preprocessing import preprocess_recording
from classifiers import LDAClassifier, SVMClassifier, RFClassifier
from train_model import train_model, create_model_info
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
from matplotlib import pyplot as plt
import mne

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def score_loocv(recording_name, constructor):
    model_info = create_model_info(recording_name)
    extractor = CSPExtractor()
    extractor.load_model(recording_name)
    X, y = preprocess_recording(recording_name, model_info)
    X_transformed = extractor.fit_transform(X, y)

    looc = LeaveOneOut()
    looc.get_n_splits(X_transformed)
    scores = []
    for train_index, test_index in looc.split(X_transformed):
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
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


def test_model(recording_name, model_name, model_constructor):
    extractor = CSPExtractor()
    extractor.load_model(model_name)
    predictor = model_constructor()
    predictor.load_model(model_name)
    model_info = predictor.model.model_info

    X, y = preprocess_recording(recording_name, model_info)
    X_transformed = extractor.fit_transform(X, y)
    acc = predictor.score(X_transformed, y)
    return np.round(acc, 3)


def test_heatmap(participants, constructor):
    data = np.zeros((len(participants) + 1, len(participants)))
    for row, model_name in enumerate(participants):
        train_model(model_name, constructor)
        for column, recording_name in enumerate(participants):
            data[row, column] = test_model(recording_name, model_name, constructor)

    for column, recording_name in enumerate(participants):
        data[len(participants), column] = score_loocv(recording_name, constructor)

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
        yticklabels=participants + ["LOOCV"],
    )
    plt.tight_layout()
    plt.title(f"{constructor.__name__} Accuracy")
    plt.xlabel("Recordings")
    plt.ylabel("Models")
    return map


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/participants")
    participants = sorted([path.stem for path in dir.glob("*.hdf5")])

    classifiers = [LDAClassifier, SVMClassifier, RFClassifier]
    for classifier in classifiers:
        heatmap = test_heatmap(participants, classifier)
        plt.savefig(f"{classifier.__name__}_accuracy_results.png")
