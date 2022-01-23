import numpy as np
from scipy.signal import iirfilter, filtfilt

# delete with DemoExtractor
from random import randint
from numpy.random import normal


def bandpass(
    trials,
    sample_rate,
    low=8,
    high=15,
):
    """Takes a 3D array of (time points x channels x trials) and filters out frequency content below the 'low' or above the 'high' threshold."""
    a, b = iirfilter(6, [low / (sample_rate / 2.0), high / (sample_rate / 2.0)])

    # check if axis=0 is correct
    if len(trials.shape) < 3:
        return filtfilt(a, b, trials, axis=0)
    else:
        filtered = np.zeros(trials.shape)
        for trial in range(trials.shape[2]):
            filtered[:, :, trial] = filtfilt(a, b, trials[:, :, trial], axis=0)
        return filtered


def cov(trials):
    n_samples = trials.shape[0]
    n_trials = trials.shape[2]
    covs = [
        trials[:, :, trial].dot(trials[:, :, trial].T) / n_samples
        for trial in range(n_trials)
    ]
    return np.mean(covs, axis=0)


def whitening(cov):
    U, l, _ = np.linalg.svd(cov)
    return U.dot(np.diag(l ** -0.5))


class DemoExtractor:
    def extract(self, data):
        return randint(0, 2) + normal(0, 0.2, 1)


class CspExtractor:
    def __init__(self, sampling_rate, models_path):
        self.sampling_rate = sampling_rate
        self.model_path = (
            models_path + "csp/model1.npy"
        )  # Do this cleanly with pathlib and env vars

    def train(self, trials_per_label):
        # takes a list of length labels containing ndarrays (time points x channels x trials)
        filtered = [
            self.bandpass(trials, self.sampling_rate) for trials in trials_per_label
        ]
        covs = [cov(trials) for trials in trials_per_label]
        P = whitening(sum(covs))  # check if this summing is correct
        B, _, _ = np.linalg.svd(
            P.T.dot(covs[-1]).dot(P)
        )  # covs[-1] is probably not correct
        W = P.dot(B)
        np.save(self.model_path, W)

    def test(self, trials):
        filtered = bandpass(trials, self.sampling_rate)
        W = np.load(self.model_path)  # Do this via env vars
        return W.T.dot(filtered)  # Check if this should be transposed
