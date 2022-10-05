import numpy as np
import sys
from pathlib import Path
import joblib

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]

from feature_extraction.extractors import CSPExtractor
from sklearn.model_selection import LeaveOneOut
from feature_extraction.extractors import CSPExtractor
from classification.classifiers import CLASSIFIERS
from data_acquisition.preprocessing import preprocess_openbci


def create_config(
    model_name="",
    description="",
    data_set="training",
    model_type="LDA",
    clf_specific={},
    channels=["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"],
    n_channels=8,
    max_trials=None,
    n_classes=3,
    imagery_window=4,
    csp_components=8,
    bandpass=(8, 13),
    notch=(50),
    notch_width=0.5,
):
    return {
        "model_name": model_name,
        "description": description,
        "model_type": model_type,
        "clf_specific": clf_specific,
        "n_channels": n_channels,
        "max_trials": max_trials,
        "channels": channels,
        "n_classes": n_classes,
        "imagery_window": imagery_window,
        "csp_components": csp_components,
        "bandpass": bandpass,
        "notch": notch,
        "notch_width": notch_width,
    }


def load_model(model):
    path = f"{DATA_PATH}/models/{model}.pkl"
    try:
        return joblib.load(path)
    except:
        print("Model not found.")


def save_model(extractor, predictor, config):
    predictor.model.config = config
    path = f"{DATA_PATH}/models/{config['model_name']}.pkl"
    joblib.dump((extractor, predictor), path)



def loocv(X, y, config):
    looc = LeaveOneOut()
    looc.get_n_splits(X)
    y_true = []
    y_pred = []

    for train_index, test_index in looc.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        extractor = CSPExtractor(config)
        X_train_transformed = extractor.fit_transform(X_train, y_train)
        X_test_transformed = extractor.transform(X_test)

        clf = CLASSIFIERS[config["model_type"]](config)
        clf.fit(X_train_transformed, y_train)
        y_true.append(int(y_test))
        y_pred.append(int(clf.predict(X_test_transformed)))

    return y_true, y_pred



def train_model(users, config, save=False, subset_idx=None):
    if isinstance(users, list):
        X, y = preprocess_openbci(users[0], config)
        for u in users:
            if u != users[0]:
                new_X, new_y = preprocess_openbci(u, config)
                X = np.append(X, new_X, axis=0)
                y = np.append(y, new_y, axis=0)
    else:
        X, y = preprocess_openbci(users, config)
        if subset_idx:
            X = X[subset_idx, :, :]
            y = y[subset_idx]

    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X, y)
    classifier = CLASSIFIERS[config["model_type"]](config)
    classifier.fit(X_transformed, y)

    if save:
        save_model(extractor, classifier, config)

    return extractor, classifier


def test_model(test_user, model, subset_idx=None, score=True):
    if isinstance(model, str):
        extractor, predictor = load_model(model)
    else:
        extractor, predictor = model

    X, y = preprocess_openbci(test_user, predictor.model.config)
    if subset_idx:
        X, y = X[subset_idx, :, :], y[subset_idx]
    X_transformed = extractor.transform(X)
    if score:
        acc = predictor.score(X_transformed, y)
        return acc
    else:
        preds = predictor.predict(X_transformed)
        return preds