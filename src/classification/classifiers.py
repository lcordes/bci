from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]


class Classifier:
    def __init__(self, type="LDA"):
        self.type = type
        if type == "LDA":
            self.model = LinearDiscriminantAnalysis()
        else:
            raise Exception("Classifier type does not exist.")

    def load_model(self, model_name):
        path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
        try:
            self.model = joblib.load(path)
        except:
            print("Classifier model not found.")

    def save_model(self, model_name):
        if self.model:
            path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
            joblib.dump(self.model, path)
        else:
            print("No fitted model found.")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_probs(self, X):
        probs = self.model.predict_proba(X)[0]
        probs = [np.round(prob, 3) for prob in probs]
        assert len(probs) == 3, "Class probabilities are not equal to 3"
        return probs
