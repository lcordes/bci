import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from natsort import natsorted
from dotenv import load_dotenv
import numpy as np
from classification.train_test_model import create_config
from classification.classifiers import CLASSIFIERS
from analysis.run_loocv_grid_search import score_loocv
from data_acquisition.preprocessing import preprocess_recording
import numpy as np
from sklearn.metrics import accuracy_score
from random import sample
import mne
mne.set_log_level(verbose="ERROR")
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def get_trials_subset(onsets, labels, max_trials):
    subset_idx = [] 
    for label in set(labels):
        label_indices = [i for i in range(len(labels)) if labels[i] == label]
        subset_idx += sample(label_indices, max_trials)
    
    return [onsets[i] for i in subset_idx], [labels[i] for i in subset_idx]




def preprocess_recording(user, config):
    assert 2 <= config["n_classes"] <= 4, "Invalid number of classes" 
    path = f"{DATA_PATH}/recordings/competition_IV_2a/{user}.gdf"

    # Load gdf file into raw data structure and filter data 
    raw = mne.io.read_raw_gdf(path, preload=True)
    channel_dict = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
    'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5':'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1',
    'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3',
    'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1',
    'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz',
     'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right'}

    raw.rename_channels(channel_dict)
    raw.drop_channels(["EOG-left","EOG-central","EOG-right"])

    if config["n_channels"] == 3:
        raw.pick_channels(["C3", "C4", "Cz"])
    elif config["n_channels"] == 7:
        raw.pick_channels(['CP1', 'C3', 'FC1', 'Cz', 'FC2', 'C4', 'CP2'])
    elif config["n_channels"] == 9:
        raw.pick_channels(['CP1', 'C3', 'FC1', 'Cz', 'FC2', 'C4', 'CP2', 'Fz', 'Pz'])

    raw2 = raw.copy()
    raw.filter(l_freq=config["bandpass"][0], h_freq=config["bandpass"][1])

    # Extract label info
    raw_onsets = raw._raw_extras[0]["events"][1]
    raw_labels = raw._raw_extras[0]["events"][2]
    onsets, labels = [], []
    for i, label in enumerate(raw_labels):

        if 769 <= label <= (768+config["n_classes"]):
            onsets.append(raw_onsets[i])
            labels.append(label)

    labels = [l-768 for l in labels]   

    if config["max_trials"]:
        onsets, labels = get_trials_subset(onsets, labels, config["max_trials"])

    # Epoch data
    event_arr = np.column_stack(
        (onsets, np.zeros(len(labels), dtype=int), labels)
    )
    epochs = mne.Epochs(raw, event_arr, tmin=0.5, tmax=3.5, baseline=None, preload=True)
    
   

    X = epochs.get_data()
    y = epochs.events[:, 2]

    return X, y

def run_config(config, users):
    accs = []
    for user in users:
        X, y = preprocess_recording(user, config)
        y_true, y_pred = score_loocv(X, y, config)
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)

    return accs



if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
    users = natsorted([path.stem for path in dir.glob("*.gdf") if path.stem[-1] == "T"]) # Only loads test sessions (have label info)
    #users = [users[0]]

    configs = [
        create_config(clf_specific={"solver": "eigen", "shrinkage": "auto"}, bandpass=(8, 30)),
        create_config(clf_specific={"solver": "eigen", "shrinkage": "auto"}, bandpass=(8, 25)),
            ]
    for c in configs:
        print("Parameters", c)
        accs = run_config(c, users)
        print(f"Mean acc: {np.mean(accs):.3f} (SD = {np.std(accs):.3f})")



    # for max_trials in [20, 30, 45]:

    #     config = create_config(max_trials=max_trials, clf_specific={"solver": "eigen", "shrinkage": "auto"}, bandpass=(8, 25))

    #     print(config)

    #     mean_accs, stds = [], []
    #     for _ in range(10):
    #         accs = run_config(config, users)
    #         print(f"Mean acc: {np.mean(accs):.3f} (SD = {np.std(accs):.3f})")
    #         mean_accs.append(np.mean(accs))
    #         stds.append(np.std(accs))

    #     print(f"Overall Mean acc: {np.mean(mean_accs):.3f} (Overall mean SD = {np.mean(stds):.3f})\n")

