from time import sleep
import os
from dotenv import load_dotenv
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np

load_dotenv()
SERIAL_PORT = os.environ["SERIAL_PORT"]
BOARD_ID = int(os.environ["BOARD_ID"])

BoardShim.enable_dev_board_logger()


class OpenBCIHandler:
    def __init__(self, sim=False):
        self.sim = sim
        self.board_id = BoardIds.SYNTHETIC_BOARD.value if self.sim else BOARD_ID
        params = BrainFlowInputParams()
        if not self.sim:
            params.serial_port = SERIAL_PORT

        self.board = BoardShim(self.board_id, params)
        self.n_trial_samples = 2 * self.get_sampling_rate()
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.status = "connected"
        except:
            self.status = "no_connection"

    def get_current_data(self, n_samples=None):
        if not n_samples:
            n_samples = self.n_trial_samples
        return self.board.get_current_board_data(n_samples).T

    def insert_marker(self, marker):
        self.board.insert_marker(marker)

    def get_sampling_rate(self):
        return self.board.get_sampling_rate(self.board_id)

    def save_and_exit(self, session_type="Live"):
        session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()
        np.save(f"data/recordings/{session_type}_session_{session_time}.", data)


if __name__ == "__main__":
    handler = OpenBCIHandler(sim=True)
    sleep(1)
    handler.save_and_exit()
