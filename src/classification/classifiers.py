from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]


class Classifier:
    def __init__(self):
        self.model = None
        self.type = "default"

    def load_model(self, model_name):
        path = f"{DATA_PATH}/models/{self.type}/{model_name}.pkl"
        try:
            self.model = joblib.load(path)
        except:
            print("Classifier model not found.")
        self.type = self.model.config["model_type"]

    def save_model(self, config):
        config["model_type"] = self.type
        self.model.config = config
        name = config["model_name"]
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
    def __init__(self, config):
        self.type = "LDA"
        self.model = LinearDiscriminantAnalysis()
        self.model.config = config


class RFClassifier(Classifier):
    def __init__(self, config):
        self.type = "RF"
        self.model = RandomForestClassifier()
        self.model.config = config


class SVMClassifier(Classifier):
    def __init__(self, config):
        self.type = "SVM"
        self.model = SVC(probability=True)
        self.model.config = config


class MLPClassifier(Classifier):
    def __init__(self, config):
        self.type = "MLP"
        self.model = MLP(max_iter=500)
        self.model.config = config
