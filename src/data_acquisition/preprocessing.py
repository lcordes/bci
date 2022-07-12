import numpy as np
import os
import mne
import h5py
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim
import matplotlib.pyplot as plt
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

plt.style.use("fivethirtyeight")
from matplotlib.lines import Line2D


load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
PRACTICE_END_MARKER = int(os.environ["PRACTICE_END_MARKER"])
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])


def get_data(recording_name, n_channels):
    # TODO: Add additional channels to raw as non-data channels

    """
    Loads a training session recording of shape (channels x samples).
    Returns a training data set of shape (trials x channels x samples)
    and a corresponding set of labels with shape (trials).
    """
    path = f"{DATA_PATH}/recordings/participants/{recording_name}.hdf5"
    # trials = np.load(path)
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
    marker_data = np.where(
        marker_data == TRIAL_END_MARKER, 0, marker_data
    )  # TODO use marker 9 as trial end when creating epochs, create check that practice trial are done correctly (marker 8)
    eeg_data = trials[eeg_channels, :]
    channel_names = list(metadata["channel_names"])

    if n_channels < eeg_data.shape[0]:
        eeg_data = eeg_data[:n_channels, :]
        channel_names = channel_names[:8]

    return eeg_data, marker_data, channel_names, sampling_rate


def raw_from_array(data, sampling_rate, channel_names):
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(data, info)
    return raw


def filter_raw(raw, bandpass=(0.1, 60), notch=(50), notch_width=50):
    raw.notch_filter(notch, notch_widths=notch_width)
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    return raw


def epochs_from_raw(raw, marker_data, sampling_rate, tmax):
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices].astype(int)
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)

    event_arr = np.column_stack(
        (onsets, np.zeros(len(marker_labels), dtype=int), marker_labels)
    )
    epochs = mne.Epochs(raw, event_arr, tmin=0, tmax=tmax, baseline=None, preload=True)
    return epochs


def preprocess_recording(recording_name, config):
    data, marker_data, channel_names, sampling_rate = get_data(
        recording_name, config["n_channels"]
    )
    raw = raw_from_array(data, sampling_rate, channel_names)
    raw = raw.pick(config["channels"])

    filtered = filter_raw(
        raw.copy(),
        bandpass=config["bandpass"],
        notch=config["notch"],
        notch_width=config["notch_width"],
    )
    epochs = epochs_from_raw(
        filtered,
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
        bar_dat[:, label - 1] = np.mean(
            log_var[labels == label, :], axis=0
        )  # check if axis correct

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
    dir = Path(f"{DATA_PATH}/recordings/participants")
    users = sorted([path.stem for path in dir.glob("*.hdf5")])

    for recording_name in users:
        bandpass = (0.5, 40)
        notch = (25, 50)
        data, marker_data, channel_names, sampling_rate = get_data(
            recording_name, n_channels=8
        )
        data = data * 1e-6
        raw = raw_from_array(
            data, sampling_rate=sampling_rate, channel_names=channel_names
        )

        filtered = filter_raw(
            raw.copy(), bandpass=bandpass, notch=notch, notch_width=0.5
        )

        marker_indices = np.argwhere(marker_data).flatten()
        marker_labels = marker_data[marker_indices].astype(int)
        onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)

        event_arr = np.column_stack(
            (onsets, np.zeros(len(marker_labels), dtype=int), marker_labels)
        )
        # filtered.plot(
        #     events=event_arr,
        #     event_color=({1: "r", 2: "g", 3: "b"}),
        #     block=True,
        #     duration=10,
        #     start=100,
        #     scalings={"eeg": 2e-4},
        # )

        event_id = {"1": 1, "2": 2, "3": 3}
        epochs = epochs_from_raw(filtered, marker_data, sampling_rate=sampling_rate)
        col_dict = {"1": "r", "2": "g", "3": "b"}
        epoch_colors = [[col_dict[str(marker)]] * 8 for marker in marker_labels]
        mne.viz.plot_epochs(
            epochs,
            epoch_colors=epoch_colors,
            block=True,
            scalings={"eeg": 2e-4},
        )

        # railed_trials_count(epochs.get_data())

        # # Look at filtering effect
        # raw.plot_psd()
        # filtered.plot_psd()

        # # Visualize power spectrum between labels
        # channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
        # plot_psd_per_label(epochs, channels, freq_window=bandpass)
        # plt.show()

        # # Visualize log variance between labels
        # channels = ["CP1", "C3", "FC1", "Cz", "FC2", "C4", "CP2", "Fpz"]
        # plot_log_variance(epochs, channels)
        # plt.show()
