from mne.decoding import CSP


class Extractor:
    def __init__(self):
        self.type = "default"
        self.model = None

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
        self.model = CSP(n_components=config["csp_components"], reg=config["csp_reg"])
