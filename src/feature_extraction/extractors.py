import joblib
import os
from dotenv import load_dotenv

from mne import create_info
from mne.decoding import CSP

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]


class Extractor:
    def __init__(self):
        self.type = "default"
        self.model = None

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
        self.model.n_channels = X.shape[1]

    def fit_transform(self, X, y):
        transform = self.model.fit_transform(X, y)
        self.model.n_channels = X.shape[1]
        return transform

    def transform(self, X):
        return self.model.transform(X)

    def get_n_channels(self):
        if not self.model:
            print("Model not loaded")
            return None
        else:
            return self.model.n_channels


class CSPExtractor(Extractor):
    def __init__(self, config=None, space=None):
        self.type = "CSP"
        self.model = (
            CSP(n_components=config["csp_components"])
            if config
            else CSP(n_components=8)
        )
        if space:
            self.model.set_params(
                transform_into=space
            )  # TODO look into space transform, else delete

    def visualize_csp(self, model_name):
        channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
        info = create_info(ch_names=channels, sfreq=125, ch_types="eeg")
        info.set_montage("standard_1020")
        path = f"{DATA_PATH}/plots/{model_name}"
        self.model.plot_patterns(info).savefig(f"{path}_csp_patterns.png")
        self.model.plot_filters(info).savefig(f"{path}_csp_filters.png")
