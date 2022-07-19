import numpy as np
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from feature_extraction.extractors import Extractor, CSPExtractor
from classification.classifiers import Classifier
from data_acquisition.preprocessing import preprocess_recording


def create_config(
    model_name="",
    channels=["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"],
    n_channels=8,
    n_classes=3,
    imagery_window=4,
    csp_components=8,
    bandpass=(8, 13),
    notch=(25, 50),
    notch_width=0.5,
):
    return {
        "model_name": model_name,
        "n_channels": n_channels,
        "channels": channels,
        "n_classes": n_classes,
        "imagery_window": imagery_window,
        "csp_components": csp_components,
        "bandpass": bandpass,
        "notch": notch,
        "notch_width": notch_width,
    }


def train_model(users, constructor, config, save=False):
    if isinstance(users, list):
        X, y = preprocess_recording(users[0], config)
        for u in users:
            if u != users[0]:
                new_X, new_y = preprocess_recording(u, config)
                X = np.append(X, new_X, axis=0)
                y = np.append(y, new_y, axis=0)
    else:
        X, y = preprocess_recording(users, config)
    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X, y)
    classifier = constructor(config)
    classifier.fit(X_transformed, y)

    if save:
        extractor.save_model(config["model_name"])
        classifier.save_model(config)

    return extractor, classifier


def test_model(test_user, model):
    if isinstance(model, str):
        extractor = Extractor()
        extractor.load_model(model)
        predictor = Classifier()
        predictor.load_model(model)
    else:
        extractor, predictor = model
    X, y = preprocess_recording(test_user, predictor.model.config)
    X_transformed = extractor.transform(X)
    acc = predictor.score(X_transformed, y)
    return acc
