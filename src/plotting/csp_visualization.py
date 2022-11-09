import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.preprocessing import preprocess_recording, get_users
from pipeline.utilities import create_config
from pipeline.feature_extraction import CSPExtractor
from pipeline.transfer_learning import get_align_mat, align
import numpy as np
import mne
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])
RESULTS_PATH = os.environ["RESULTS_PATH"]


def csp_visualization(config, user, align_X=False, save=False, rm_outliers=True):    
    if user == "all":
        users = get_users(config)
        config["discard_railed"] = False
    else:
        users = [user]

    X_all, y_all = [], []
    for u in users:
        X, y = preprocess_recording(u, config)
        if align_X:
            align_mat = get_align_mat(X)
            X = align(X, align_mat)     
        X_all.append(X)
        y_all.extend(y)

    X_all = np.concatenate(X_all, axis=0)

    extractor = CSPExtractor(config)
    extractor.fit(X_all, y_all)
    info = mne.create_info(ch_names=config['channels'], sfreq=125, ch_types="eeg")
    info.set_montage("standard_1020")
    extractor.model.plot_patterns(info, colorbar=False)

    if save:
        aligned = "_aligned" if align_X else ""
        title = f"csp_{config['data_set']}_{user}{aligned}"
        plt.savefig(f"{RESULTS_PATH}/data_exploration/{title}.png", dpi=400)
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    # for data_set in ["training", "evaluation", "benchmark"]:
    #     for align_X in [False, True]:
    #         config = create_config({"data_set": data_set, "discard_railed": False})
    #         csp_visualization(config, user="all", align_X=align_X, save=True)

    config = create_config({"data_set": "evaluation", "discard_railed": False})
    csp_visualization(config, user="E06", align_X=False, save=True)

    # config = create_config({"data_set": "evaluation", "discard_railed": True,
    #  "csp_components": 6, "channels": ["C3", "Cz", "FC2", "C4", "CP2", "Fpz"]})
    # csp_visualization(config, user="E06", align_X=False, save=True)

