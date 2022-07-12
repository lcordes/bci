import argparse
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition.preprocessing import preprocess_recording
from classifiers import LDAClassifier, RFClassifier, SVMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import os
from dotenv import load_dotenv

import matplotlib
import mne

matplotlib.use("Agg")
mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def create_config(
    model_name="",
    channels=["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"],
    n_channels=8,
    n_classes=3,
    imagery_window=2,
    bandpass=(8, 13),
    notch=(25, 50),
    notch_width=0.5,
):
    return {
        "name": model_name,
        "n_channels": n_channels,
        "channels": channels,
        "n_classes": n_classes,
        "imagery_window": imagery_window,
        "bandpass": bandpass,
        "notch": notch,
        "notch_width": notch_width,
    }


def train_model(recording, constructor, config):
    if isinstance(recording, list):
        config["name"] = "merged_model"
        X, y = preprocess_recording(recording[0], config)
        for r in recording:
            if r != recording[0]:
                new_X, new_y = preprocess_recording(r, config)
                X = np.append(X, new_X, axis=0)
                y = np.append(y, new_y, axis=0)
    else:
        config["name"] = recording
        X, y = preprocess_recording(recording, config)
    extractor = CSPExtractor()
    X_transformed = extractor.fit_transform(X, y)
    extractor.save_model(config["name"])
    classifier = constructor()
    classifier.fit(X_transformed, y)
    classifier.save_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recording_name",
        help="Name of the recording to train from (without extension).",
    )
    parser.add_argument(
        "model_name", help="Name for saving the trained model (without extension)."
    )
    parser.add_argument(
        "--daisy",
        help="Use all 16 channels for model training",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    recording_name = args.recording_name.replace(".hdf5", "")
    model_name = args.model_name.replace(".pkl", "")

    # Set model parameters
    n_channels = 16 if args.daisy else 8
    config = create_config(model_name, n_channels=n_channels)

    X, y = preprocess_recording(recording_name, config)
    class_frequencies = np.asarray(np.unique(y, return_counts=True))
    print("Class frequencies:\n", class_frequencies)

    # Train and save the extractor
    extractor = CSPExtractor()
    X_transformed = extractor.fit_transform(X, y)
    extractor.save_model(config["name"])
    print("Extractor trained and saved successfully.")

    # Train and save the classifier
    classifier = LDAClassifier()
    classifier.fit(X_transformed, y)
    classifier.save_model(config)
    print("Classifier trained and saved successfully.")

    from test_model import loocv_per_block

    loocv_per_block(LinearDiscriminantAnalysis, X_transformed, y)
