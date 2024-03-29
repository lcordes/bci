from natsort import natsorted
import numpy as np
import mne
from mne.filter import notch_filter, filter_data
from random import sample
import h5py
from brainflow.board_shim import BoardShim
from pathlib import Path
import sys

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
RAILED_THRESHOLD = int(os.environ["RAILED_THRESHOLD"])

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)


def get_users(config):
    dir = Path(f"{DATA_PATH}/recordings/{config['data_set']}")
    file_format = "gdf" if config['data_set'] == "benchmark" else "hdf5"
    users = natsorted([path.stem for path in dir.glob(f"*.{file_format}")])
    return users


def test_recording_sampling_rate(marker_data):
    lengths = []
    for i, label in enumerate(list(marker_data)):
        if label in [1,2,3]:
            trial_start = i
        if label == 9:
            lengths.append((i - trial_start)//5)
    print("Recording sampling rate:", np.mean(lengths))


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


def get_not_railed_channels(X, config):

    if isinstance(X, str):
        copy_config = config.copy()
        copy_config["discard_railed"] = False
        copy_config["bandpass"] = None
        copy_config["notch"] = None
        X, _ = preprocess_recording(X, copy_config)

    railed_count = railed_trials_count(X)
    not_railed_idx = np.flatnonzero(railed_count < 5)
    channel_names = [ch for i, ch in enumerate(config["channels"]) if i in not_railed_idx]
    return channel_names


def get_eval_user_id(user):
    path = f"{DATA_PATH}/recordings/evaluation/{user}.hdf5"
    with h5py.File(path, "r") as file:
        metadata = dict(file["data"].attrs)
    return metadata["participant number"]


def get_trials_subset(X, y, max_trials):
    """Return a random subset of trials in X and y (consisting of 'max_trials' trials per class)."""
    subset_idx = [] 
    for label in set(y):
        label_indices = [i for i in range(len(y)) if y[i] == label]
        subset_idx += sample(label_indices, max_trials)
    
    return X[subset_idx,:,:], y[subset_idx]

def preprocess_openbci(user, config, calibration=False):
    """Load a user recording from an openbci data set, convert data into raw format and extract
    marker events."""

    assert 2 <= config["n_classes"] <= 3, "Invalid number of classes" 
    
    # Load data
    if isinstance(user, str):
        path = f"{DATA_PATH}/recordings/{config['data_set']}/{user}.hdf5"
        with h5py.File(path, "r") as file:
            recording = file["data"][()]
            metadata = dict(file["data"].attrs)
    else:
        recording, metadata, calibration = user # used for online model training during recording

    # Get board specific info
    board_info = BoardShim.get_board_descr(metadata["board_id"])
    sampling_rate = board_info["sampling_rate"]
    eeg_channels = board_info["eeg_channels"]
    marker_channel = board_info["marker_channel"]

    # Disregard practice/calibration trials or keep only practice trials for calibration
    practice_end = np.where(recording[marker_channel, :] == PRACTICE_END_MARKER)[0][0]

    if not calibration:
        recording = recording[:, (practice_end + 1) :]
    else:
        recording = recording[:, :practice_end]


    # Extract events info
    marker_data = recording[marker_channel, :].flatten()
    onsets = np.argwhere(marker_data).flatten().astype(int)
    labels = marker_data[onsets].astype(int)
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = onsets
    events[:, 2] = labels
    events = events[events[:, 2] != TRIAL_END_MARKER, :]

    # Create raw instance
    eeg_data = recording[eeg_channels, :]
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


def preprocess_recording(user, config, calibration=False):
    """Preprocess a user's recording depending on the specific data set and config parameters."""

    if config["data_set"] in ["training", "evaluation"]:
        raw, events = preprocess_openbci(user, config, calibration)
    elif config["data_set"] == "benchmark":
        raw, events = preprocess_benchmark(user, config)

    # Pick relevant channels and filter data
    raw = raw.pick(config["channels"])

    if config["discard_railed"] and not config["data_set"] == "benchmark":
        epochs = mne.Epochs(raw, events, tmin=config["tmin"], tmax=config["tmax"], baseline=None, preload=True)
        not_railed_channels = get_not_railed_channels(epochs.get_data(), config)
        raw = raw.pick(not_railed_channels)
        railed_channels = set(config['channels']) - set(not_railed_channels)
        if len(not_railed_channels) < len(config["channels"]):
            print(f"Flatlined channels removed for {user}: {railed_channels}")

    if config["bandpass"]:
        raw.filter(l_freq=config["bandpass"][0], h_freq=config["bandpass"][1])
    if config["notch"]:
        raw.notch_filter(freqs=config["notch"], notch_widths=config["notch_width"])
   
    # Create epochs
    epochs = mne.Epochs(raw, events, tmin=config["tmin"], tmax=config["tmax"], baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Subset data depending on n_classes and max_trials parameters
    for label in set(y):
        if label > config["n_classes"]:
            X = X[y != label, :, :]
            y = y[y != label]

    if config["max_trials"]:
        X, y = get_trials_subset(X, y, config["max_trials"])
    
    return X, y


def preprocess_trial(data, sampling_rate, channel_names, config):
    """Get data of length (ONLINE_FILTER_LENGTH) seconds, with
    ONLINE_FILTER_LENGTH - IMAGERY_PERIOD giving the trial onset,
    filter it and extract the interest period."""

    # Pick channel subset
    channel_idx = [i for i, channel in enumerate(channel_names) if channel in config["channels"]]
    
    # Filter
    if config["notch"]:
        data = notch_filter(
            data, sampling_rate, freqs=config["notch"], notch_widths=config["notch_width"]
        )
    if config["bandpass"]:
        data = filter_data(
            data, sampling_rate, l_freq=config["bandpass"][0], h_freq=config["bandpass"][1]
        )

    # Epoch
    start = int(
        (ONLINE_FILTER_LENGTH - (IMAGERY_PERIOD - TRIAL_OFFSET)) * sampling_rate
    )
    end = start + config["imagery_window"] * sampling_rate
    interest_period = data[:, channel_idx, start:end]

    return interest_period

