import numpy as np


class LDA:
    def __init__(self):
        self.W = []
        self.b = []

    def fit(self, X, y):
        """
        args:
            X - A 2D array (time points * channels, maybe reverse)
            y - A 1D array containing the class labels per time point
        """
        labels = np.unique(y)
        data_per_label = [X[y == label, :] for label in labels]
        priors = np.array([(y == label).mean() for label in labels])
        means = [np.mean(label_data, axis=0) for label_data in data_per_label]
        centered = [
            label_data - mean for mean, label_data in zip(means, data_per_label)
        ]
        covariances = np.array(
            [
                center.T.dot(center / (len(data_per_label) - len(labels)))
                for center in centered
            ]
        )

        self.W = (np.mean(np.array(means) * -1)).dot(
            np.linalg.pinv((priors * covariances).sum())
        )
        self.b = (priors * np.array(means)).sum().dot(self.W)

    def predict(self, X):
        pred = round(float(X))
        if pred < 0 or pred > 2:
            pred = 2
        return pred

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass
