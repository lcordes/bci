import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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
from pipeline.preprocessing import preprocess_recording, get_users, railed_trials_count


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
        plt.savefig(f"{RESULTS_PATH}/data_exploration/railed_channels_{config['data_set']}.png", dpi=250)
    else:
        plt.show()


if __name__ == "__main__":

    config = create_config({"data_set": "evaluation", "bandpass": None, "notch": None, "discard_railed": False})
    railed_heatmap(config, save=True)
