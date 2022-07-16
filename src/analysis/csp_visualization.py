import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])


def plot_events_in_raw(raw, marker_data, sampling_rate):
    marker_indices = np.argwhere(marker_data).flatten()
    marker_labels = marker_data[marker_indices].astype(int)
    onsets = (marker_indices + sampling_rate * TRIAL_OFFSET).astype(int)

    event_arr = np.column_stack(
        (onsets, np.zeros(len(marker_labels), dtype=int), marker_labels)
    )
    raw.plot(
        events=event_arr,
        event_color=({1: "r", 2: "g", 3: "b"}),
        block=True,
        duration=10,
        start=100,
        scalings={"eeg": 2e-4},
    )


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
