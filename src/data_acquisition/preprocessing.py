import numpy as np
import os
import mne
from mne.filter import notch_filter, filter_data
import h5py
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim
import sys
from pathlib import Path
from random import sample

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
IMAGERY_PERIOD = float(os.environ["IMAGERY_PERIOD"])
PRACTICE_END_MARKER = int(os.environ["PRACTICE_END_MARKER"])
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])


def get_data(recording_name, n_channels):
    """
    Loads a training session recording of shape (channels x samples).
    Returns a training data set of shape (trials x channels x samples)
    and a corresponding set of labels with shape (trials).
    """
    path = f"{DATA_PATH}/recordings/training_data_collection/{recording_name}.hdf5"
    with h5py.File(path, "r") as file:
        trials = file["data"][()]
        metadata = dict(file["data"].attrs)

    assert (
        trials.shape[0] < trials.shape[1]
    ), "Data shape incorrect, there are more channels than samples."

    # Get board specific info
    board_id = metadata["board_id"]
    assert board_id in [-1, 2], "Invalid board_id in recording"
    board_info = BoardShim.get_board_descr(board_id)
    sampling_rate = board_info["sampling_rate"]
    eeg_channels = board_info["eeg_channels"]
    marker_channel = board_info["marker_channel"]

    # Disregard practice trials
    practice_end = np.where(trials[marker_channel, :] == PRACTICE_END_MARKER)[0][0]
    trials = trials[:, (practice_end + 1) :]

    marker_data = trials[marker_channel, :].flatten()
    marker_data = np.where(marker_data == TRIAL_END_MARKER, 0, marker_data)
    eeg_data = trials[eeg_channels, :]
    channel_names = list(metadata["channel_names"])

    if n_channels < eeg_data.shape[0]:
        eeg_data = eeg_data[:n_channels, :]
        channel_names = channel_names[:n_channels]

    return eeg_data, marker_data, channel_names, sampling_rate


def raw_from_array(data, sampling_rate, channel_names):
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(data, info)
    return raw


def filter_array(array, sampling_rate, bandpass, notch, notch_width=0.5):
    array = notch_filter(
        array, sampling_rate, freqs=notch, notch_widths=notch_width, verbose="WARNING"
    )
    array = filter_data(
        array, sampling_rate, l_freq=bandpass[0], h_freq=bandpass[1], verbose="WARNING"
    )
    return array


def epochs_from_raw(raw, marker_data, sampling_rate, tmax):
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices].astype(int)
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)

    event_arr = np.column_stack(
        (onsets, np.zeros(len(marker_labels), dtype=int), marker_labels)
    )
    epochs = mne.Epochs(raw, event_arr, tmin=0, tmax=tmax, baseline=None, preload=True)
    return epochs

def get_trials_subset(X, y, max_trials):
    subset_idx = [] 
    for label in set(y):
        label_indices = [i for i in range(len(y)) if y[i] == label]
        subset_idx += sample(label_indices, max_trials)
    
    return X[subset_idx,:,:], y[subset_idx]


def preprocess_openbci(user, config):
    assert 2 <= config["n_classes"] <= 3, "Invalid number of classes" 

    data, marker_data, channel_names, sampling_rate = get_data(
        user, config["n_channels"]
    )

    if config["bandpass"]:
        filtered = filter_array(
            data,
            sampling_rate,
            bandpass=config["bandpass"],
            notch=config["notch"],
            notch_width=config["notch_width"],
        )
    else:
        filtered = data

    raw = raw_from_array(filtered, sampling_rate, channel_names)
    raw = raw.pick(config["channels"])

    epochs = epochs_from_raw(
        raw,
        marker_data,
        sampling_rate=sampling_rate,
        tmax=config["imagery_window"],
    )
    X = epochs.get_data()
    y = epochs.events[:, 2]

    if config["n_classes"] == 2:
        X = X[y != 3, :, :]
        y = y[y != 3]

    if config["max_trials"]:
        X, y = get_trials_subset(X, y, config["max_trials"])

    return X, y

def get_trials_subset_benchmark(onsets, labels, max_trials):
    subset_idx = [] 
    for label in set(labels):
        label_indices = [i for i in range(len(labels)) if labels[i] == label]
        subset_idx += sample(label_indices, max_trials)
    
    return [onsets[i] for i in subset_idx], [labels[i] for i in subset_idx]




def preprocess_benchmark(user, config):
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
        onsets, labels = get_trials_subset_benchmark(onsets, labels, config["max_trials"])

    # Epoch data
    event_arr = np.column_stack(
        (onsets, np.zeros(len(labels), dtype=int), labels)
    )
    epochs = mne.Epochs(raw, event_arr, tmin=0.5, tmax=3.5, baseline=None, preload=True)
    
    X = epochs.get_data()
    y = epochs.events[:, 2]

    return X, y

def preprocess_recording(user, config):
    if config["data_set"] in ["training", "evaluation"]:
        X, y = preprocess_openbci(user, config)
    elif config["data_set"] == "benchmark":
        X, y = preprocess_benchmark(user, config)
    

def preprocess_trial(data, sampling_rate, config):
    """Get data of length (ONLINE_FILTER_LENGTH) seconds, with
    ONLINE_FILTER_LENGTH - IMAGERY_PERIOD giving the trial onset,
    filter it and extract the interest period."""
    filtered = filter_array(
        data,
        sampling_rate,
        bandpass=config["bandpass"],
        notch=config["notch"],
        notch_width=config["notch_width"],
    )
    start = int(
        (ONLINE_FILTER_LENGTH - (IMAGERY_PERIOD - TRIAL_OFFSET)) * sampling_rate
    )
    end = start + config["imagery_window"] * sampling_rate
    interest_period = filtered[:, :, start:end]

    return interest_period

