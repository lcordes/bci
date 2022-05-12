import argparse
import numpy as np
import json
from time import sleep
import mne
from matplotlib import pyplot as plt

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_acquisition.data_handler import OpenBCIHandler


def check_railed(data):
    """Take a data array of shape (channels x samples) and check whether channels are railed
    (i.e. values stay the same for more than half of the samples). Returns two lists containing the indices of railed and
    non-railed channels."""
    railed = []
    not_railed = []
    # Get nums of railed hannels
    for channel in range(data.shape[0]):
        railed_ratio = 1 - (len(np.unique(data[channel, :])) / data.shape[1])
        if railed_ratio < 0.5:
            railed.append(channel)
        else:
            not_railed.append(channel)

    return railed, not_railed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--board",
        dest="board_type",
        choices=["synthetic", "cython", "daisy"],
        default="daisy",
        const="daisy",
        nargs="?",
        type=str,
        help="Use synthetic or cython board instead of daisy.",
    )

    parser.add_argument(
        "--n_channels",
        choices=[8, 16],
        const=16,
        default=16,
        nargs="?",
        type=str,
        help="Choose number of channels to investigate for railing",
    )

    parser.add_argument(
        "--plot",
        help="Generate live head plot of railed channels",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    handler = OpenBCIHandler(board_type=args.board_type)
    with open("architecture/channel_map.json", "r") as file:
        channel_map = json.load(file)
    check_window = 4  # time window in s in which railing is determined

    if args.plot:
        info = mne.create_info(
            ch_names=list(channel_map.values()),
            sfreq=handler.sampling_rate,
            ch_types="eeg",
        )
        info.set_montage("standard_1020")
        fig = mne.viz.plot_sensors(info, pointsize=100, show=False)
        plt.pause(0.2)

    while True:
        n_samples = handler.sampling_rate * check_window
        sleep(check_window)
        sample = np.squeeze(
            handler.get_current_data(args.n_channels, n_samples=n_samples)
        )
        railed, not_railed = check_railed(sample)

        if railed:
            names = [channel_map[str(num + 1)] for num in railed]
            print(f"Potentially railed channels: {names}")
        else:
            print("Nothing looks railed.")

        if args.plot:
            plt.close()
            fig = mne.viz.plot_sensors(
                info,
                show_names=True,
                ch_groups=[[], not_railed, railed],
                pointsize=100,
                show=False,
                linewidth=0,
            )
            plt.pause(0.2)
