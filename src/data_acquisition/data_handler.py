from time import sleep
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
from random import randint
import json
import os
from dotenv import load_dotenv
import h5py

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
CHANNEL_MAP_PATH = os.environ["CHANNEL_MAP_PATH"]


import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

BoardShim.enable_dev_board_logger()


def get_channel_map():
    with open(CHANNEL_MAP_PATH, "r") as file:
        channel_map = json.load(file)
    return channel_map


class OpenBCIHandler:
    def __init__(self, board_type, recording_name=None):
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
        self.trial = 1
        self.session_id = randint(100000, 999999)
        self.channel_map = get_channel_map()
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"
        self.session_start = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    def get_current_data(self, n_channels, n_samples=None):
        n_samples = (
            int(self.info["sampling_rate"] * TRIAL_LENGTH)
            if not n_samples
            else n_samples
        )
        data = self.board.get_current_board_data(n_samples)

        # Disregard first row (time sample_channel) and then keep the next n_channel rows
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

    def save_trial(self):
        """Get and remove current data from the buffer after every trial"""
        data = self.board.get_board_data()
        np.save(
            f"{DATA_PATH}/recordings/tmp/#{self.session_id}_trial_{self.trial}", data
        )
        self.trial += 1

    def combine_trial_data(self):
        """Merge individual trial data into one file and add descriptives"""
        for trial in range(1, self.trial):
            data = np.load(
                f"{DATA_PATH}/recordings/tmp/#{self.session_id}_trial_{trial}.npy"
            )

            # Append trial and board_id info to recording
            trial_row = np.full((1, data.shape[1]), trial)
            board_id_row = np.full((1, data.shape[1]), self.board_id)
            data = np.append(data, trial_row, axis=0)
            data = np.append(data, board_id_row, axis=0)
            if trial == 1:
                recording = data
            else:
                recording = np.append(recording, data, axis=1)
        return recording

    def get_metadata(self):
        metadata = {}
        metadata["board_id"] = self.board_id
        metadata["session_id"] = self.session_id
        metadata["session_start"] = self.session_start
        metadata["session_end"] = self.session_end
        metadata["channel_names"] = list(self.channel_map.values())
        return metadata

    def save_to_file(self, recording):
        if self.recording_name:
            file_path = f"{DATA_PATH}/recordings/{self.recording_name}"
        else:
            file_path = f"{DATA_PATH}/recordings/Training_session_#{self.session_id}_{self.session_end}"

        with h5py.File(f"{file_path}.hdf5", "w") as file:
            d = file.create_dataset("data", data=recording)
            d.attrs.update(self.get_metadata())

    def merge_trials_and_exit(self):
        """Stop data stream and attempt to merge trial data files"""
        self.board.stop_stream()
        self.board.release_session()
        self.session_end = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        try:
            recording = self.combine_trial_data()
            self.save_to_file(recording)

            # TODO: delete tmp files here?
            print(f"Successfully aggregated session #{self.session_id} recording file.")
        except Exception as e:
            print("Couldn't aggregate tmp file, got error:", e)


if __name__ == "__main__":
    handler = OpenBCIHandler(board_type="synthetic")

    for i in range(5):
        handler.insert_marker(i + 1)
        sleep(1)
        handler.save_trial()

    handler.merge_trials_and_exit()
