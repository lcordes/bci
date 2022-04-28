# %%
import numpy as np
import os
import mne
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)


load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def get_data(recording_name, cython=False):
    # TODO: Instead of deleting additional channels add them to raw as non-data channels

    """
    Loads a training session recording of shape (channels x samples).
    Returns a training data set of shape (trials x channels x samples)
    and a corresponding set of labels with shape (trials).
    """
    path = f"{DATA_PATH}/recordings/{recording_name}.npy"
    trials = np.load(path)
    assert (
        trials.shape[0] < trials.shape[1]
    ), "Data shape incorrect, there are more channels than samples."

    # Get board specific info
    board_id = int(trials[-1, -1])
    assert board_id in [-1, 2], "Invalid board_id in recording"
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # Extract marker info
    sample_channel = 0
    marker_channel = BoardShim.get_marker_channel(board_id)
    board_channel = trials.shape[0] - 1  # Last channel/row
    assert marker_channel in [
        31,
        17,
    ], "Check if marker channel is correct in prepare_trials"
    marker_data = trials[marker_channel, :].flatten()

    # Remove non-voltage channels
    trials_cleaned = np.delete(
        trials, [sample_channel, marker_channel, board_channel], axis=0
    )
    trials_cleaned = trials_cleaned[:8, :] if cython else trials_cleaned[:16, :]
    return trials_cleaned, marker_data, sampling_rate


def raw_from_npy(data, sampling__rate):
    channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
    info = mne.create_info(ch_names=channels, sfreq=sampling__rate, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(data, info)
    return raw


def filter_raw(raw, bandpass=(0.1, 60), notch=(50), notch_width=50):
    raw.notch_filter(notch, notch_widths=notch_width)
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    return raw


def create_epochs(filtered, marker_data, sampling_rate):
    # Rewrite this to

    # Extract trials
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)
    samples_per_trial = int((sampling_rate * TRIAL_LENGTH))
    ends = onsets + samples_per_trial

    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices]
    assert set(marker_labels).issubset({1.0, 2.0, 3.0}), "Labels are incorrect."
    train_data = np.zeros((len(marker_labels), filtered.shape[0], samples_per_trial))

    for i in range(len(marker_labels)):
        train_data[i, :, :] = filtered[:, onsets[i] : ends[i]]

    assert (
        train_data.shape[2] == sampling_rate * TRIAL_LENGTH
    ), "Number of samples is incorrect"
    return train_data, marker_labels


# %%

model_name = "real"
data, markers, sampling_rate = get_data(model_name, cython=True)
raw = raw_from_npy(data, sampling__rate=sampling_rate)
filtered = filter_raw(raw.copy(), notch=(25, 50), notch_width=0.5)
filtered.plot_psd()
# %% Alternative

path = f"{DATA_PATH}/recordings/real.npy"
trials = np.load(path)

raw = raw_from_npy(trials[1:9, :], sampling__rate=125)
filtered = filter_raw(raw.copy(), notch=(25, 50), notch_width=0.5)

# Plots
# filtered.plot_psd()
# raw.plot_psd_topo(fmin=8, fmax=13)
# %%
