from time import sleep
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np

import os
from dotenv import load_dotenv

from classification.classifiers import DATA_PATH

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])
TRIAL_OFFSET = float(os.environ["TRIAL_OFFSET"])

BoardShim.enable_dev_board_logger()


class OpenBCIHandler:
    def __init__(self, board_type):
        self.board_type = board_type
        params = BrainFlowInputParams()
        if board_type == "synthetic":
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        elif board_type in ["cython", "daisy"]:
            # Cython and Cython daisy board both get daisy parameters, as the headset
            # is still streaming all 16 channels with 125 sampling rate
            self.board_id = BoardIds.CYTON_DAISY_BOARD.value
            params.serial_port = SERIAL_PORT
        else:
            raise Exception("OpenBCIHandler got unsupported board type")

        self.board = BoardShim(self.board_id, params)
        self.marker_channel = self.board.get_marker_channel(self.board_id)
        self.sample_channel = 0
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"

        self.sr = self.board.get_sampling_rate(self.board_id)

    def select_channels(self, data):
        # Make sure these channels are indeed the 8 cython and 8 daisy channels!!
        if self.board_type == "cython":
            return data[:8, :]
        else:
            return data[:16, :]

    def get_current_data(self, n_samples=None):

        if not n_samples:
            n_samples = int(self.sr * (TRIAL_OFFSET + TRIAL_LENGTH))
        data = self.board.get_current_board_data(n_samples)
        assert (
            self.marker_channel == data.shape[0] - 1
        ), "Marker channel isn't last column in current data, something off?"

        data = np.delete(data, [self.sample_channel, self.marker_channel], axis=0)
        offset_end = int(TRIAL_OFFSET * self.sr)
        data = data[:, offset_end:]

        data = self.select_channels(data)
        data = np.expand_dims(data, axis=0)
        return data

    def get_concentration_data(self, n_samples=None):
        if not n_samples:
            n_samples = int(self.sr * 5)
        return self.board.get_current_board_data(n_samples)

    def insert_marker(self, marker):
        self.board.insert_marker(marker)

    def get_sampling_rate(self):
        return self.board.get_sampling_rate(self.board_id)

    def get_board_id(self):
        return self.board_id

    def save_and_exit(self, session_type="Live"):
        session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        data = self.board.get_board_data()
        # Append column with board_id to recording
        board_id_row = np.full((1, data.shape[1]), self.board_id)
        data = np.append(data, board_id_row, axis=0)
        self.board.stop_stream()
        self.board.release_session()
        np.save(f"{DATA_PATH}/recordings/{session_type}_session_{session_time}", data)


if __name__ == "__main__":
    handler = OpenBCIHandler(sim=True)
    sleep(1)
    handler.save_and_exit()
