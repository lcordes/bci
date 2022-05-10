from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
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
        assert len(probs) == 3, "Class probabilities are not equal to 3"
        return probs

    def score_looc(self, X, y):
        looc = LeaveOneOut()
        looc.get_n_splits(X)
        scores = []
        for train_index, test_index in looc.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = self.model_constructor()
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
        print("Leave-one-out-CV mean accuracy:", np.round(np.mean(scores), 3))


class LDAClassifier(Classifier):
    def __init__(self):
        self.type = "LDA"
        self.model_constructor = LinearDiscriminantAnalysis
        self.model = self.model_constructor()
