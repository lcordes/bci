import numpy as np
import os
import mne
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def epochs_from_raw(raw, marker_data, sampling_rate):
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices].astype(int)
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)

    event_arr = np.column_stack(
        (onsets, np.zeros(len(marker_labels), dtype=int), marker_labels)
    )
    print(event_arr[:5, :])
    epochs = mne.Epochs(raw, event_arr, tmin=0, tmax=2, baseline=None, preload=True)
    return epochs


def plot_psd_per_label(epochs, channels, freq_window):
    fig, ax = plt.subplots(len(channels))
    label_cols = ["red", "green", "blue"]
    for channel_idx, channel in enumerate(channels):
        for label_idx, label in enumerate(["1", "2", "3"]):
            epochs[label].plot_psd(
                ax=ax[channel_idx],
                color=label_cols[label_idx],
                picks=[channel],
                fmin=freq_window[0],
                fmax=freq_window[1],
                show=False,
                spatial_colors=False,
            )
        ax[channel_idx].set_title(channel)

    fig.set_tight_layout(True)
    fig.suptitle("Power spectral density per class")
    custom_lines = [Line2D([0], [0], color=col, lw=2) for col in label_cols]
    fig.legend(custom_lines, ["left", "right", "up"])

    plt.show()


if __name__ == "__main__":
    model_name = "real"
    data, marker_data, sampling_rate = get_data(model_name, cython=True)
    raw = raw_from_npy(data, sampling__rate=sampling_rate)
    filtered = filter_raw(raw.copy(), bandpass=(1, 60), notch=(25, 50), notch_width=0.5)
    epochs = epochs_from_raw(filtered, marker_data, sampling_rate=sampling_rate)

    # Look at filtering effect
    # raw.plot_psd()
    # filtered.plot_psd()

    # Visualize power spectrum between labels
    channels = ["C3", "Cz", "C4"]
    plot_psd_per_label(epochs, channels, freq_window=(8, 13))
