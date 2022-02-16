from mne.decoding import CSP
import joblib
import numpy as np
import os
from dotenv import load_dotenv

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter
from brainflow.exit_codes import *
from brainflow.ml_model import (
    MLModel,
    BrainFlowMetrics,
    BrainFlowClassifiers,
    BrainFlowModelParams,
)

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def prepare_trials(
    recording_name,
    sampling_rate=250,
    marker_channel=31,
    sample_channel=0,
):
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

    # Extract marker info
    marker_data = trials[marker_channel, :].flatten()
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices]
    assert set(marker_labels).issubset({1.0, 2.0, 3.0}), "Labels are incorrect."

    # Remove non-voltage channels
    trials_cleaned = np.delete(trials, [sample_channel, marker_channel], axis=0)
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


class MIExtractor:
    def __init__(self, type="CSP", space=None):
        self.type = type
        if type == "CSP":
            self.model = CSP(n_components=30)
            if space:
                self.model.set_params(
                    transform_into=space
                )  # get optional **params to work here
        else:
            raise Exception("Extractor type does not exist.")

    def load_model(self, model_name):
        path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
        try:
            self.model = joblib.load(path)
        except:
            print("Extractor model not found.")

    def save_model(self, model_name):
        if self.model:
            path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
            joblib.dump(self.model, path)
        else:
            print("No fitted model found.")

    def fit(self, X, y):
        self.model.fit(X, y)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    def transform(self, X):
        return self.model.transform(X)


class ConcentrationExtractor:
    def __init__(self, sr, board_id):
        self.sr = sr
        self.board_id = board_id

    def get_concentration(self, data):
        eeg_channels = BoardShim.get_eeg_channels(int(self.board_id))
        data = np.squeeze(data)
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, self.sr, True)
        feature_vector = np.concatenate((bands[0], bands[1]))

        # calc concentration
        concentration_params = BrainFlowModelParams(
            BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value
        )
        concentration = MLModel(concentration_params)
        concentration.prepare()
        conc = int(concentration.predict(feature_vector) * 100)
        concentration.release()

        return conc


if __name__ == "__main__":
    X, y = prepare_trials("data/recordings/train_example.npy")
    print(X.shape, y.shape)
