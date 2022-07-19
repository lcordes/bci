import numpy as np
import os
import mne
from mne.filter import notch_filter, filter_data
import h5py
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

mne.set_log_level("WARNING")

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
IMAGERY_PERIOD = float(os.environ["IMAGERY_PERIOD"])
PRACTICE_END_MARKER = int(os.environ["PRACTICE_END_MARKER"])
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])
ONLINE_FILTER_LENGTH = os.environ["ONLINE_FILTER_LENGTH"]


def get_data(recording_name, n_channels):
    """
    Loads a training session recording of shape (channels x samples).
    Returns a training data set of shape (trials x channels x samples)
    and a corresponding set of labels with shape (trials).
    """
    path = f"{DATA_PATH}/recordings/users/{recording_name}.hdf5"
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
    # TODO this gives 501 samples, make congruent with online samples
    return epochs


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


def preprocess_recording(recording_name, config):
    data, marker_data, channel_names, sampling_rate = get_data(
        recording_name, config["n_channels"]
    )

    filtered = filter_array(
        data,
        sampling_rate,
        bandpass=config["bandpass"],
        notch=config["notch"],
        notch_width=config["notch_width"],
    )

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

    return X, y
