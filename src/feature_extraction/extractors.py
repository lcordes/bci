import joblib
import numpy as np
import os
from dotenv import load_dotenv

from mne import create_info
from mne.decoding import CSP
from matplotlib import pyplot as plt

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


class MIExtractor:
    def __init__(self, type="CSP", space=None):
        self.type = type
        if type == "CSP":
            self.model = CSP(n_components=4)
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
            print((self.model.n_channels))
            path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
            joblib.dump(self.model, path)
        else:
            print("No fitted model found.")

    def fit(self, X, y):
        self.model.fit(X, y)
        self.model.n_channels = X.shape[1]
        assert self.model.n_channels in [
            8,
            16,
            30,
        ], "Extractor got invalid channel size"

    def fit_transform(self, X, y):
        transform = self.model.fit_transform(X, y)
        self.model.n_channels = X.shape[1]
        assert self.model.n_channels in [
            8,
            16,
            30,
        ], "Extractor got invalid channel size"
        return transform

    def transform(self, X):
        return self.model.transform(X)

    def get_n_channels(self):
        if not self.model:
            print("Model not yrtloaded")
            return None
        else:
            return self.model.n_channels

    def visualize_csp(self, model_name):
        channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
        info = create_info(ch_names=channels, sfreq=125, ch_types="eeg")
        info.set_montage("standard_1020")
        self.model.plot_patterns(info)
        path = f"{DATA_PATH}/plots/csp_patters_{model_name}.png"
        plt.savefig(path)


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
