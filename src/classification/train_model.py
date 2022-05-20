import argparse
import sys
from pathlib import Path
from tracemalloc import start

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import CSPExtractor
from data_acquisition import preprocessing as pre
from classifiers import LDAClassifier
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
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


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
    recording_name = args.recording_name.replace(".pkl", "")
    model_name = args.model_name.replace(".pkl", "")

    # Preprocess the trial data
    n_channels = 16 if args.daisy else 8
    data, marker_data, channel_names, sampling_rate = pre.get_data(
        recording_name, n_channels
    )
    raw = pre.raw_from_array(data, sampling_rate, channel_names)
    bandpass = (8, 13)
    notch = (25, 50)
    filtered = pre.filter_raw(
        raw.copy(), bandpass=bandpass, notch=notch, notch_width=0.5
    )
    epochs = pre.epochs_from_raw(filtered, marker_data, sampling_rate=sampling_rate)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    class_frequencies = np.asarray(np.unique(y, return_counts=True))
    print("Class frequencies:\n", class_frequencies)
    channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
    # pre.save_preprocessing_plots(model_name, channels, raw, filtered, epochs, bandpass)

    # Train and save the extractor
    extractor = CSPExtractor()
    X_transformed = extractor.fit_transform(X, y)
    extractor.save_model(model_name)
    # extractor.visualize_csp(model_name)
    print("Extractor trained and saved successfully.")
    # pre.plot_log_variance_2(X_transformed, y).savefig(
    #   "comp_var.png"
    # )  # TODO Is this plot necessary?

    # Train and save the classifier
    classifier = LDAClassifier()
    classifier.fit(X_transformed, y)
    classifier.save_model(model_name)
    print("Classifier trained and saved successfully.")

    overall_acc = classifier.score_loocv(X_transformed, y)
    print(X_transformed.shape, y.shape)
    print(f"Overall loocv mean accuracy: {overall_acc}")

    for block in range(1, 4):
        block_len = len(y) // 3
        start = (block - 1) * block_len
        stop = block * block_len
        block_acc = classifier.score_loocv(X_transformed[start:stop, :], y[start:stop])
        print(f"Block {block} loocv mean accuracy: {block_acc}")
