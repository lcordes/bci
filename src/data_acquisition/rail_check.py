import argparse
import numpy as np
from time import sleep
import mne
from matplotlib import pyplot as plt

import sys
from pathlib import Path
import os

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from data_acquisition.data_handler import OpenBCIHandler, get_channel_map
from matplotlib.animation import FuncAnimation

RAILED_THRESHOLD = 100000  # 187000


def check_railed(data):
    """Take a data array of shape (channels x samples) and check whether channels are railed
    (i.e. values stay the same for more than half of the samples). Returns two lists containing the indices of railed and
    non-railed channels."""
    railed = []
    not_railed = []
    # Get nums of railed hannels
    for channel in range(data.shape[0]):
        railed_ratio = np.mean(np.abs(data[channel, :]) > RAILED_THRESHOLD)
        # railed_ratio = 1 - (len(np.unique(data[channel, :])) / data.shape[1])
        if railed_ratio > 0.5:
            railed.append(channel)
        else:
            not_railed.append(channel)

    return not_railed, railed


def update_plot(frame):
    sample = np.squeeze(handler.get_current_data(args.n_channels, n_samples=n_samples))
    not_railed, railed = check_railed(sample)

    if railed:
        names = [channel_map[str(num + 1)] for num in railed]
        print(f"Potentially railed channels: {names}")

    fig = mne.viz.plot_sensors(
        info,
        show_names=True,
        axes=ax,
        ch_groups=[[], not_railed, railed],
        pointsize=100,
        show=False,
        linewidth=0,
    )
    plt.pause(0.2)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--board",
        dest="board_type",
        choices=["synthetic", "cyton", "daisy"],
        default="daisy",
        const="daisy",
        nargs="?",
        type=str,
        help="Use synthetic or cyton board instead of daisy.",
    )

    parser.add_argument(
        "--n_channels",
        choices=[8, 16],
        default=8,
        nargs="?",
        type=int,
        help="Choose number of channels to investigate for railing",
    )

    args = parser.parse_args()

    channel_map = get_channel_map()
    if args.n_channels == 8:
        channel_map = {
            key: value for key, value in channel_map.items() if int(key) <= 8
        }
    handler = OpenBCIHandler(board_type=args.board_type)
    check_window = 2  # update interval in ms
    sampling_rate = handler.get_sampling_rate()
    n_samples = sampling_rate * check_window
    info = mne.create_info(
        ch_names=list(channel_map.values()),
        sfreq=sampling_rate,
        ch_types="eeg",
    )
    info.set_montage("standard_1020")
    sleep(check_window)

    fig = plt.figure()
    ax = fig.add_subplot()
    interval = check_window * 1000
    anim = FuncAnimation(fig=fig, func=update_plot, interval=interval)
    plt.show()
