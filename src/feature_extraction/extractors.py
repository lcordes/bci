from mne.decoding import CSP

import joblib
import os
from dotenv import load_dotenv

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

    def fit_transform(self, X, y):
        transform = self.model.fit_transform(X, y)
        return transform

    def transform(self, X):
        return self.model.transform(X)


class CSPExtractor(Extractor):
    def __init__(self, config):
        self.type = "CSP"

        self.model = CSP(n_components=config["csp_components"])
