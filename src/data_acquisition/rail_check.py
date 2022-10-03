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

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from classification.train_test_model import create_config
from data_acquisition.preprocessing import preprocess_recording

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


def railed_heatmap(users, ax, config):
    railed_dat = np.zeros((len(users), config["n_channels"]))
    for i, name in enumerate(users):
        X, y = preprocess_recording(name, config)
        railed_dat[i, :] = railed_trials_count(X)
    # Show the tick labels
    ax.xaxis.set_tick_params(labeltop=True)

    # Hide the tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    x_ticks = config["channels"]
    labels = [u.replace("Training_session_", "")[:7] for u in users]

    sns.heatmap(
        railed_dat,
        ax=ax,
        annot=True,
        fmt="g",
        xticklabels=x_ticks,
        yticklabels=labels,
    )
    ax.set_title("Number of railed trials per recording per channel")


if __name__ == "__main__":

    dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])
    config = create_config(clf_specific={"shrinkage": "auto", "solver": "eigen"}, bandpass=None)
    fig = plt.figure()
    ax = fig.add_subplot()
    railed_heatmap(users,ax, config)
    plt.savefig("railed.png", dpi=300)
