import numpy as np
import os
import mne
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
from matplotlib.lines import Line2D


load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def get_data(recording_name, n_channels):
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
    # TODO: update assert (trial_channel is now addtional column in recordings)
    # assert marker_channel in [
    #     31,
    #     17,
    # ], "Check if marker channel is correct in prepare_trials"
    marker_data = trials[marker_channel, :].flatten()

    # Remove non-voltage channels
    trials_cleaned = np.delete(
        trials, [sample_channel, marker_channel, board_channel], axis=0
    )
    trials_cleaned = trials_cleaned[:n_channels, :]
    return trials_cleaned, marker_data, sampling_rate


def raw_from_npy(data, sampling__rate):
    channels = [
        "CP1",
        "C3",
        "FC1",
        "Cz",
        "FC2",
        "C4",
        "CP2",
        "Fpz",
    ]  # TODO Double check order, also add option for 16 channel
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
    epochs = mne.Epochs(raw, event_arr, tmin=0, tmax=2, baseline=None, preload=True)
    return epochs


def plot_psd_per_label(epochs, channels, freq_window):
    fig, ax = plt.subplots(3, 3)
    label_cols = ["red", "green", "blue"]
    for channel_idx, channel in enumerate(channels):
        for label_idx, label in enumerate(["1", "2", "3"]):
            epochs[label].plot_psd(
                ax=fig.axes[channel_idx],
                color=label_cols[label_idx],
                picks=[channel],
                fmin=freq_window[0],
                fmax=freq_window[1],
                show=False,
                spatial_colors=False,
            )
        fig.axes[channel_idx].set_title(channel)

    fig.set_tight_layout(True)
    fig.suptitle("Power spectral density per class")
    custom_lines = [Line2D([0], [0], color=col, lw=2) for col in label_cols]
    fig.legend(custom_lines, ["left", "right", "up"])
    return fig


def plot_log_variance(epochs, channels):
    # Get bar data
    epochs_subset = epochs.pick(channels)
    dat = epochs_subset.get_data()
    log_var = np.log(np.var(dat, axis=2))  # correct to take log var here and not later?
    labels = epochs_subset.events[:, 2]

    bar_dat = np.zeros((dat.shape[1], 3))
    for label in [1, 2, 3]:
        bar_dat[:, label - 1] = np.mean(log_var[labels == label, :], axis=0)

    fig, ax = plt.subplots(3, 3, sharey=True)
    for idx, channel in enumerate(channels):
        fig.axes[idx].bar(["left", "right", "up"], bar_dat[idx, :])
        cutoff = np.min(bar_dat) - 0.2
        fig.axes[idx].set_ylim(bottom=cutoff)

        fig.axes[idx].set_title(channel)

    fig.set_tight_layout(True)
    fig.suptitle("Log variance per class")
    return fig


def plot_log_variance_2(transformed, y):

    label_dat = [transformed[y == label, :] for label in [1, 2, 3]]
    label_dat = [np.log(np.var(dat, axis=0)) for dat in label_dat]
    fig, ax = plt.subplots(3, 3, sharey=True)
    for idx in range(8):
        bar_dat = [label_dat[0][idx], label_dat[1][idx], label_dat[2][idx]]
        fig.axes[idx].bar(["left", "right", "up"], bar_dat)
    fig.set_tight_layout(True)
    fig.suptitle("Log variance per class")
    return fig


def save_preprocessing_plots(name, channels, raw, filtered, epochs, bandpass):
    path = f"{DATA_PATH}/plots/{name}"
    raw.plot_psd().savefig(f"{path}_psd_unfiltered.png")
    filtered.plot_psd().savefig(f"{path}_psd_filtered.png")
    plot_psd_per_label(epochs, channels, freq_window=bandpass).savefig(
        f"{path}_psd_per_class.png"
    )
    plot_log_variance(epochs, channels).savefig(f"{path}_log_variance_per_class.png")


if __name__ == "__main__":
    recording_name = "real"
    bandpass = (8, 13)
    notch = (25, 50)
    data, marker_data, sampling_rate = get_data(recording_name, n_channels=8)
    raw = raw_from_npy(data, sampling__rate=sampling_rate)
    filtered = filter_raw(raw.copy(), bandpass=bandpass, notch=notch, notch_width=0.5)
    epochs = epochs_from_raw(filtered, marker_data, sampling_rate=sampling_rate)

    # Look at filtering effect
    raw.plot_psd()
    filtered.plot_psd()

    # Visualize power spectrum between labels
    channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
    plot_psd_per_label(epochs, channels, freq_window=bandpass)
    plt.show()

    # Visualize log variance between labels
    channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
    plot_log_variance(epochs, channels)
    plt.show()
