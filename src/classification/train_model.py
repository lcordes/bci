import argparse
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from feature_extraction.extractors import MIExtractor
from classifiers import Classifier
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def prepare_trials(recording_name, cython=False):
    """
    Loads a training session recording of shape (channels x samples).
    Returns a training data set of shape (trials x channels x samples)
    and a corresponding set of labels with shape (trials).
    """
    path = f"{DATA_PATH}/recordings/{recording_name}.npy"
    trials = np.load(path)
    assert (
        trials.shape[0] < trials.shape[1]
    ), "Data shape incorrect, there are more channels than samples."

    # Get board specific info
    board_id = int(trials[-1, -1])
    assert board_id in [-1, 2], "Invalid board_id in recording"
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # Extract marker info
    sample_channel = 0
    marker_channel = BoardShim.get_marker_channel(board_id)
    board_channel = trials.shape[0] - 1  # Last channel/row
    assert marker_channel in [
        31,
        17,
    ], "Check if marker channel is correct in prepare_trials"
    marker_data = trials[marker_channel, :].flatten()
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices]
    assert set(marker_labels).issubset({1.0, 2.0, 3.0}), "Labels are incorrect."

    # Remove non-voltage channels
    trials_cleaned = np.delete(
        trials, [sample_channel, marker_channel, board_channel], axis=0
    )
    trials_cleaned = trials_cleaned[:8, :] if cython else trials_cleaned[:16, :]

    # Extract trials
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)
    samples_per_trial = int((sampling_rate * TRIAL_LENGTH))
    ends = onsets + samples_per_trial

    train_data = np.zeros(
        (len(marker_labels), trials_cleaned.shape[0], samples_per_trial)
    )

    for i in range(len(marker_labels)):
        train_data[i, :, :] = trials_cleaned[:, onsets[i] : ends[i]]

    assert (
        train_data.shape[2] == sampling_rate * TRIAL_LENGTH
    ), "Number of samples is incorrect"
    return train_data, marker_labels


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
        "--cython",
        help="Train model on the 8 cython channels only.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    recording_name = args.recording_name.replace(".pkl", "")
    model_name = args.model_name.replace(".pkl", "")

    # Prepare the trial data
    X, y = prepare_trials(recording_name, cython=args.cython)
    class_frequencies = np.asarray(np.unique(y, return_counts=True))
    print("Class examples:\n", class_frequencies)

    # Train and save the extractor
    extractor = MIExtractor(type="CSP")
    X_transformed = extractor.fit_transform(X, y)
    extractor.save_model(model_name)
    extractor.visualize_csp(model_name)
    print("Extractor trained and saved successfully.")

    # Train and save the extractor
    classifier = Classifier(type="LDA")
    classifier.fit(X_transformed, y)
    classifier.save_model(model_name)
    print("Classifier trained and saved successfully.")
