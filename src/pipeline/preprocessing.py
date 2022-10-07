from natsort import natsorted
import numpy as np
import mne
from mne.filter import notch_filter, filter_data
import h5py
from brainflow.board_shim import BoardShim
from random import sample
from pathlib import Path

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
mne.set_log_level(verbose="ERROR")

import os
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
IMAGERY_PERIOD = float(os.environ["IMAGERY_PERIOD"])
PRACTICE_END_MARKER = int(os.environ["PRACTICE_END_MARKER"])
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])


def get_users(config):
    dir = Path(f"{DATA_PATH}/recordings/{config['data_set']}")
    file_format = "gdf" if config['data_set'] == "benchmark" else "hdf5"
    users = natsorted([path.stem for path in dir.glob(f"*.{file_format}")])
    return users

def get_trials_subset(X, y, max_trials):
    """Return a random subset of trials in X and y (consisting of 'max_trials' trials per class)."""
    subset_idx = [] 
    for label in set(y):
        label_indices = [i for i in range(len(y)) if y[i] == label]
        subset_idx += sample(label_indices, max_trials)
    
    return X[subset_idx,:,:], y[subset_idx]


def preprocess_openbci(user, config):
    """Load a user recording from an openbci data set, convert data into raw format and extract
    marker events."""

    assert 2 <= config["n_classes"] <= 3, "Invalid number of classes" 
    
    # Load data
    path = f"{DATA_PATH}/recordings/{config['data_set']}/{user}.hdf5"
    with h5py.File(path, "r") as file:
        trials = file["data"][()]
        metadata = dict(file["data"].attrs)

    # Get board specific info
    board_info = BoardShim.get_board_descr(metadata["board_id"])
    sampling_rate = board_info["sampling_rate"]
    eeg_channels = board_info["eeg_channels"]
    marker_channel = board_info["marker_channel"]

    # Disregard practice trials
    practice_end = np.where(trials[marker_channel, :] == PRACTICE_END_MARKER)[0][0]
    trials = trials[:, (practice_end + 1) :]

    # Extract events info
    marker_data = trials[marker_channel, :].flatten()
    onsets = np.argwhere(marker_data).flatten().astype(int)
    labels = marker_data[onsets].astype(int)
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = onsets
    events[:, 2] = labels
    events = events[events[:, 2] != TRIAL_END_MARKER, :]

    # Create raw instance
    eeg_data = trials[eeg_channels, :]
    channel_names = list(metadata["channel_names"])
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(eeg_data, info)

    return raw, events
   

def preprocess_benchmark(user, config):
    """Load a user recording from the benchmark data set, convert data into raw format and extract
    marker events."""
    assert 2 <= config["n_classes"] <= 4, "Invalid number of classes" 

    # Load gdf file into raw data structure
    path = f"{DATA_PATH}/recordings/{config['data_set']}/{user}.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True)
    channel_dict = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
    'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5':'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1',
    'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3',
    'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1',
    'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz',
     'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right'}
    raw.rename_channels(channel_dict)
    
    # Extract label info
    onsets = np.array(raw._raw_extras[0]["events"][1])
    labels = np.array(raw._raw_extras[0]["events"][2])
    events = np.zeros((len(onsets), 3), dtype=int)
    events[:, 0] = onsets
    events[:, 2] = labels
    events = events[np.logical_and(769 <= labels, labels <= 772), :]
    events[:, 2] -= 768

    return raw, events


def preprocess_recording(user, config):
    """Preprocess a user's recording depending on the specific data set and config parameters."""

    if config["data_set"] in ["training", "evaluation"]:
        raw, events = preprocess_openbci(user, config)
    elif config["data_set"] == "benchmark":
        raw, events = preprocess_benchmark(user, config)

    # Pick relevant channels and filter data
    raw = raw.pick(config["channels"])
    if config["bandpass"]:
        raw.filter(l_freq=config["bandpass"][0], h_freq=config["bandpass"][1])
    if config["notch"]:
        raw.notch_filter(freqs=config["notch"], notch_widths=config["notch_width"])
   
    # Create epochs
    tmin = TRIAL_OFFSET
    tmax = tmin + config["imagery_window"]
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Subset data depending on max_trials and n_classes parameters
    for label in set(y):
        if label > config["n_classes"]:
            X = X[y != label, :, :]
            y = y[y != label]

    if config["max_trials"]:
        X, y = get_trials_subset(X, y, config["max_trials"])

    return X, y


def preprocess_trial(data, sampling_rate, config):
    """Get data of length (ONLINE_FILTER_LENGTH) seconds, with
    ONLINE_FILTER_LENGTH - IMAGERY_PERIOD giving the trial onset,
    filter it and extract the interest period."""
    filtered = notch_filter(
        data, sampling_rate, freqs=config["notch"], notch_widths=config["notch_width"]
    )
    filtered = filter_data(
        filtered, sampling_rate, l_freq=config["bandpass"][0], h_freq=config["bandpass"][1]
    )
    start = int(
        (ONLINE_FILTER_LENGTH - (IMAGERY_PERIOD - TRIAL_OFFSET)) * sampling_rate
    )
    end = start + config["imagery_window"] * sampling_rate
    interest_period = filtered[:, :, start:end]

    return interest_period

