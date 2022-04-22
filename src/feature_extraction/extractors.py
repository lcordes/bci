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
