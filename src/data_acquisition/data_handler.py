from time import sleep
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
from random import randint
import os
from dotenv import load_dotenv

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
DATA_PATH = os.environ["DATA_PATH"]
TRIAL_LENGTH = float(os.environ["TRIAL_LENGTH"])

BoardShim.enable_dev_board_logger()


class OpenBCIHandler:
    def __init__(self, board_type):
        self.board_type = board_type
        params = BrainFlowInputParams()
        if board_type == "synthetic":
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        elif board_type == "cython":
            self.board_id = BoardIds.CYTON_BOARD.value
            params.serial_port = SERIAL_PORT
        elif board_type == "daisy":
            self.board_id = BoardIds.CYTON_DAISY_BOARD.value
            params.serial_port = SERIAL_PORT
        else:
            raise Exception("OpenBCIHandler got unsupported board type")

        self.board = BoardShim(self.board_id, params)
        self.sample_channel = 0
        self.sampling_rate = self.board.get_sampling_rate(self.board_id)
        self.trial = 1
        self.session_id = randint(100000, 999999)
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"

    def get_current_data(self, n_channels):
        n_samples = int(self.sampling_rate * TRIAL_LENGTH)
        data = self.board.get_current_board_data(n_samples)

        # Disregard first row (time sample_channel) and then keep the next n_channel rows
        data = data[1 : (n_channels + 1), :]
        data = np.expand_dims(data, axis=0)
        return data

    def insert_marker(self, marker):
        self.board.insert_marker(marker)

    def get_sampling_rate(self):
        return self.board.get_sampling_rate(self.board_id)

    def get_board_id(self):
        return self.board_id

    def save_and_exit(self):
        """Get all session data from buffer and save to file"""
        session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()

        # Append board_id info to recording
        board_id_row = np.full((1, data.shape[1]), self.board_id)
        data = np.append(data, board_id_row, axis=0)

        np.save(f"{DATA_PATH}/recordings/Live_session_{session_time}", data)

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

    def merge_trials_and_exit(self):
        """Stop data stream and attempt to merge trial data files"""
        self.board.stop_stream()
        self.board.release_session()

        try:
            session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            recording = self.combine_trial_data()
            np.save(
                f"{DATA_PATH}/recordings/Training_session_#{self.session_id}_{session_time}",
                recording,
            )
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
