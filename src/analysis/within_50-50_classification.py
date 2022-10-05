
import itertools
import numpy as np
from scipy.linalg import inv, sqrtm
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from seaborn import heatmap

import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from random import sample
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from classification.train_test_model import create_config
from data_acquisition.preprocessing import preprocess_openbci as preprocess_training
from analysis.benchmark_analysis import preprocess_openbci as preprocess_benchmark
from feature_extraction.extractors import CSPExtractor
from classification.classifiers import CLASSIFIERS
from sklearn.model_selection import train_test_split


def train_model(X, y, config):
    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X, y)
    classifier = CLASSIFIERS[config["model_type"]](config)
    classifier.fit(X_transformed, y)
    return extractor, classifier


def test_model(X, y, model):
    extractor, predictor = model
    X_transformed = extractor.transform(X)
    acc = predictor.score(X_transformed, y)
    return acc


def process_data_set(data_set, config):
    if data_set == "training":
        dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
        users = natsorted([path.stem for path in dir.glob("*.hdf5")])
        preprocess_recording = preprocess_training

    elif data_set == "benchmark":
        dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
        users = natsorted([path.stem for path in dir.glob("*.gdf") if path.stem[-1] == "T"])
        preprocess_recording = preprocess_benchmark
    X_all, y_all = [], []

    for user in users:
        X, y = preprocess_recording(user, config)
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all, users


def within_classification(data_set, config, title):
    X_all, y_all, users = process_data_set(data_set, config)
    n_trials = X_all[0].shape[0]
    repetitions = 100
    results = np.zeros((len(users), repetitions))

    for u, user in enumerate(users):
        print(f"User {user}")
        for rep in range(repetitions):
            train_idx, test_idx = train_test_split(list(range(n_trials)), train_size=0.5, stratify=y_all[u])
            X_train = X_all[u][train_idx, :, :]
            X_test = X_all[u][test_idx, :, :]
            y_train = y_all[u][train_idx]
            y_test = y_all[u][test_idx]

            model = train_model(X_train, y_train, config)
            acc = test_model(X_test, y_test, model)
            results[u, rep] = acc

    print("User_results:", np.mean(results, axis=1))
    print("Average:", np.mean(np.mean(results, axis=1)))
    with open(f"{RESULTS_PATH}/within/{title}.npy", 'wb') as f:
        np.save(f, results)



if __name__ == "__main__":
    data_set = "benchmark"
    config = create_config(clf_specific={"shrinkage": "auto", "solver": "eigen"}, bandpass=(8, 30))
    title = "Within classification 50-50 split (benchmark data, 8-30, 100 reps)"
    within_classification(data_set, config, title)
    