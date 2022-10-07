from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC, NuSVC


class Classifier:
    def __init__(self):
        self.model = None
        self.type = "default"

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
        self.model = LinearDiscriminantAnalysis(**config["clf_specific"])
        self.model.config = config


class QDAClassifier(Classifier):
    def __init__(self, config):
        self.type = "QDA"
        self.model = QuadraticDiscriminantAnalysis(**config["clf_specific"])
        self.model.config = config


class RFClassifier(Classifier):
    def __init__(self, config):
        self.type = "RF"
        self.model = RandomForestClassifier(**config["clf_specific"])
        self.model.config = config


class SVMClassifier(Classifier):
    def __init__(self, config):
        self.type = "SVM"
        self.model = SVC(**config["clf_specific"])
        self.model.config = config


class NUSVMClassifier(Classifier):
    def __init__(self, config):
        self.type = "NUSVM"
        self.model = NuSVC(**config["clf_specific"])
        self.model.config = config


class MLPClassifier(Classifier):
    def __init__(self, config):
        self.type = "MLP"
        self.model = MLP(**config["clf_specific"])
        self.model.config = config


CLASSIFIERS = {
    "RF": RFClassifier,
    "SVM": SVMClassifier,
    "NUSVM": NUSVMClassifier,
    "LDA": LDAClassifier,
    "QDA": QDAClassifier,
    "MLP": MLPClassifier,
}
