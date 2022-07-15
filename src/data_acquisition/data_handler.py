from time import sleep
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
from random import randint, choice
import json
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_acquisition.preprocessing import get_data
import h5py

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])
CHANNEL_MAP_PATH = os.environ["CHANNEL_MAP_PATH"]
SERIAL_PORT = os.environ["SERIAL_PORT"]

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

BoardShim.enable_dev_board_logger()


def get_channel_map():
    with open(CHANNEL_MAP_PATH, "r") as file:
        channel_map = json.load(file)
    return channel_map


class RecordingHandler:
    def __init__(self, recording_name, config):
        self.config = config
        self.recording, marker_data, _, self.sampling_rate = get_data(
            recording_name, config["n_channels"]
        )
        self.trial_onsets = {}
        for num, label in zip([1, 2, 3], ["left", "right", "down"]):
            self.trial_onsets[label] = list(
                np.argwhere(marker_data == num).flatten().astype(int)
            )

    def get_current_data(self, label):
        trial_half = self.sampling_rate * 5  # 10s of data for filter
        onset = choice(self.trial_onsets[label])
        data = self.recording[:, (onset - trial_half) : (onset + trial_half)]
        data = np.expand_dims(data, axis=0)
        return data

    def get_sampling_rate(self):
        return self.sampling_rate


class OpenBCIHandler:
    def __init__(self, board_type, recording_name=None):
        self.clean_tmp()
        self.board_type = board_type
        self.recording_name = recording_name
        params = BrainFlowInputParams()
        if board_type == "synthetic":
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        elif board_type == "cyton":
            self.board_id = BoardIds.CYTON_BOARD.value
            params.serial_port = SERIAL_PORT
        elif board_type == "daisy":
            self.board_id = BoardIds.CYTON_DAISY_BOARD.value
            params.serial_port = SERIAL_PORT
        else:
            raise Exception("OpenBCIHandler got unsupported board type")

        self.board = BoardShim(self.board_id, params)
        self.info = self.board.get_board_descr(self.board_id)
        self.save_num = 0
        self.session_id = randint(100000, 999999)
        self.channel_map = get_channel_map()
        self.metadata = {}
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"
        self.session_start = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def get_current_data(self, n_channels):
        n_samples = int(self.info["sampling_rate"] * 10)
        data = self.board.get_current_board_data(n_samples)

        # Disregard first row (packet num channel) and then keep the next n_channel rows
        data = data[1 : (n_channels + 1), :]
        data = np.expand_dims(data, axis=0)
        return data

    def insert_marker(self, marker):
        self.board.insert_marker(marker)

    def get_sampling_rate(self):
        return self.info["sampling_rate"]

    def get_board_id(self):
        return self.board_id

    def get_channel_info(self):
        return self.board.get_board_descr(self.board_id)

    def load_demographics(self):
        try:
            with open(f"{DATA_PATH}/recordings/demographics.json", "r") as file:
                demographics = json.load(file)
                self.add_metadata(demographics)
            if "" in list(demographics.values()):
                raise Exception

            with open(f"{DATA_PATH}/recordings/demographics.json", "w") as file:
                empty_dict = {key: "" for key in demographics.keys()}
                json.dump(empty_dict, file)
            print(demographics)
        except:
            self.status = "invalid_demographics"

    def compile_metadata(self):
        self.metadata["board_id"] = self.board_id
        self.metadata["session_id"] = self.session_id
        self.metadata["session_start"] = self.session_start
        self.metadata["session_end"] = self.session_end
        self.metadata["channel_names"] = list(self.channel_map.values())

    def add_metadata(self, data):
        self.metadata.update(data)

    def save_trial(self):
        """Get and remove current data from the buffer after every trial"""
        data = self.board.get_board_data()
        self.save_num += 1
        np.save(
            f"{DATA_PATH}/recordings/tmp/#{self.session_id}_trial_{self.save_num}", data
        )

    def combine_trial_data(self):
        """Merge individual trial data into one file and add descriptives"""
        for save in range(1, self.save_num + 1):
            data = np.load(
                f"{DATA_PATH}/recordings/tmp/#{self.session_id}_trial_{save}.npy"
            )

            if save == 1:
                recording = data
            else:
                recording = np.append(recording, data, axis=1)

        if (
            "trial_sequence" in self.metadata
            and not len(self.metadata["trial_sequence"]) == self.save_num
        ):
            print("Experiment and handler trials seem to be incongruent.")
        return recording

    def save_to_file(self, recording):
        if self.recording_name:
            file_path = f"{DATA_PATH}/recordings/{self.recording_name}"
        else:
            file_path = f"{DATA_PATH}/recordings/Training_session_#{self.session_id}_{self.session_end}"

        self.compile_metadata()
        with h5py.File(f"{file_path}.hdf5", "w") as file:
            d = file.create_dataset("data", data=recording)
            d.attrs.update(self.metadata)

    def merge_trials_and_exit(self):
        """Stop data stream and attempt to merge trial data files"""
        self.board.stop_stream()
        self.board.release_session()
        self.session_end = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        if self.save_num > 0:
            try:
                recording = self.combine_trial_data()
                self.save_to_file(recording)

                # TODO: delete tmp files here?
                print(
                    f"Successfully aggregated session #{self.session_id} recording file."
                )
                print("Metadata:", self.metadata)
            except Exception as e:
                print("Couldn't aggregate tmp file, got error:", e)

    def clean_tmp(self):
        path = f"{DATA_PATH}/recordings/tmp/"
        for tmp in Path(path).glob("*.npy"):
            tmp.unlink()


if __name__ == "__main__":
    handler = OpenBCIHandler(board_type="synthetic")

    for i in range(5):
        handler.insert_marker(i + 1)
        sleep(1)
        handler.save_trial()

    handler.merge_trials_and_exit()
