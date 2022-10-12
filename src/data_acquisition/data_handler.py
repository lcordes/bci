from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
from random import randint, choice
import json
import os
import h5py
from dotenv import load_dotenv
import sys
from pathlib import Path
from time import sleep

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)
from pipeline.preprocessing import preprocess_recording
from pipeline.utilities import create_config

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
DATA_PATH = os.environ["DATA_PATH"]
CHANNEL_MAP_PATH = os.environ["CHANNEL_MAP_PATH"]
IMAGERY_PERIOD = float(os.environ["IMAGERY_PERIOD"])
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])

BoardShim.enable_dev_board_logger()


def get_channel_map():
    with open(CHANNEL_MAP_PATH, "r") as file:
        channel_map = json.load(file)
    return channel_map


class RecordingHandler:
    def __init__(self, user, config):
        self.config = config.copy()
        self.config.update({"bandpass": None, "notch": None})
        self.X, self.y = preprocess_recording(user, self.config)


    def get_current_data(self, label):
        """Return a randomly  chosen trial of the given label with shape 
        (1, n_channels, ONLINE_FILTER_LENGTH * sampling_rate), where the trial
        onset is given by  ONLINE_FILTER_LENGTH - IMAGERY_PERIOD """
        trial_idx = choice([i for i in range(self.X.shape[0]) if self.y[i] == label])
        data = self.X[trial_idx, :, :]
        return np.expand_dims(data, axis=0)
   


class OpenBCIHandler:
    def __init__(self, board_type, config):
        self.clean_tmp()
        self.board_type = board_type
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
        relevant_channels = config["channels"]
        self.channel_indices = [int(key) for key, value in self.channel_map.items() if value in relevant_channels]
        print(self.channel_indices)

        self.metadata = {}
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"
        self.session_start = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.session_end = None

    def get_current_data(self): 
        """Return the last ONLINE_FILTER_LENGTH seconds of data as an array of shape 
        (n_channels, ONLINE_FILTER_LENGTH * sampling_rate)."""
        
        n_samples = int(self.info["sampling_rate"] * ONLINE_FILTER_LENGTH)
        data = self.board.get_current_board_data(n_samples)[self.channel_indices, :]
        return np.expand_dims(data, axis=0)
        
    def insert_marker(self, marker):
        self.board.insert_marker(marker)

    def get_sampling_rate(self):
        return self.info["sampling_rate"]

    def test_real_sampling_rate(self):      
        self.insert_marker(20)
        sleep(3)
        self.insert_marker(20)
        sleep(0.5)
        data = self.board.get_board_data()
        markers = np.argwhere(data[self.info["marker_channel"], :])
        print("Sampling rate approximately", (markers[1][0] - markers[0][0]) // 3)


    def get_board_id(self):
        return self.board_id

    def get_metadata(self):
        self.compile_metadata()
        return self.metadata

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
        print("Channel config: ", self.metadata["channel_names"])

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

        saves = [np.load(
            f"{DATA_PATH}/recordings/tmp/#{self.session_id}_trial_{save}.npy") 
            for save in range(1, self.save_num + 1)
        ]
        return np.concatenate(saves, axis=1)


    def save_to_file(self, recording):
        file_path = f"{DATA_PATH}/recordings/{self.session_end}_training_session_#{self.session_id}_"

        self.compile_metadata()
        with h5py.File(f"{file_path}.hdf5", "w") as file:
            d = file.create_dataset("data", data=recording)
            d.attrs.update(self.metadata)

    def merge_trials_and_exit(self):
        """Stop data stream and attempt to merge trial data files"""
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception as e:
            print("Error releasing board session:", e)
        self.session_end = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        if self.save_num > 0:
            #try:
            recording = self.combine_trial_data()
            self.save_to_file(recording)
            print(
                f"Successfully aggregated session #{self.session_id} recording file."
            )
            #except Exception as e:
             #   print("Got error when aggregating tmp files:", e)

    def clean_tmp(self):
        path = f"{DATA_PATH}/recordings/tmp/"
        for tmp in Path(path).glob("*.npy"):
            tmp.unlink()
