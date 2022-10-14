import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from natsort import natsorted

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.utilities import create_config
from pipeline.preprocessing import preprocess_recording, get_users

RAILED_THRESHOLD = 100000 


def check_railed(data):
    """Take a data array of shape (channels x samples) and check whether channels are railed
    (i.e. values are above a threshold for more than half of the samples). Return an array of length channels
    containing zeros (channel clean) and ones (channel railed)"""
    n_channels = data.shape[0]
    channels = np.zeros(n_channels)
    for i in range(n_channels):
        railed_ratio = np.mean(np.abs(data[i, :]) > RAILED_THRESHOLD)
        if railed_ratio > 0.1:
            channels[i] = 1
    return channels


def railed_trials_count(epochs):
    n_epochs, n_channels, _ = epochs.shape
    railed = np.zeros(n_channels)
    for trial in range(n_epochs):
        railed += check_railed(epochs[trial, :, :])
    return railed


def railed_heatmap(config, save=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    users = get_users(config)
    results = np.zeros((len(users), len(config["channels"])))
    for i, name in enumerate(users):
        X, _ = preprocess_recording(name, config)
        results[i, :] = railed_trials_count(X)
    
    # Adjust tick labels
    ax.xaxis.set_tick_params(labeltop=True)
    ax.xaxis.set_tick_params(labelbottom=False)
    x_ticks = config["channels"]
    labels = [u.replace("Training_session_", "")[:7] for u in users]

    sns.heatmap(
        results,
        ax=ax,
        annot=True,
        fmt="g",
        xticklabels=x_ticks,
        yticklabels=labels,
    )
    ax.set_title("Number of railed trials per recording per channel")

    if save:
        plt.savefig(f"{RESULTS_PATH}/diagnostics/{config['data_set']}_data_railed_channels.png", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":

    config = create_config({"data_set": "evaluation", "bandpass": None, "notch": None})
    railed_heatmap(config, save=True)
