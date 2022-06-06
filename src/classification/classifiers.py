from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]


class Classifier:
    def __init__(self):
        self.model = None
        self.model_constructor = None
        self.type = "default"

    def load_model(self, model_name):
        path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
        try:
            self.model = joblib.load(path)
        except:
            print("Classifier model not found.")

    def save_model(self, model_info):
        self.model.model_info = model_info
        name = model_info["name"]
        path = f"{DATA_PATH}/models/{self.type}/{name}.pkl"
        joblib.dump(self.model, path)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict_probs(self, X):
        probs = self.model.predict_proba(X)[0]
        assert len(probs) == 3, "Class probabilities are not equal to 3"
        return probs


class LDAClassifier(Classifier):
    def __init__(self):
        self.type = "LDA"
        self.model_constructor = LinearDiscriminantAnalysis
        self.model = self.model_constructor()


class RFClassifier(Classifier):
    def __init__(self):
        self.type = "RF"
        self.model_constructor = RandomForestClassifier
        self.model = self.model_constructor()


class SVMClassifier(Classifier):
    def __init__(self):
        self.type = "SVM"
        self.model_constructor = SVC
        self.model = self.model_constructor()
