from time import sleep
import numpy as np
from datetime import datetime
from random import randint
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

BoardShim.enable_dev_board_logger()
SERIAL_PORT = ""


class OpenBCIHandler:
    def __init__(self, sim=False):
        self.sim = sim
        self.board_id = (
            BoardIds.SYNTHETIC_BOARD.value
            if self.sim
            else BoardIds.CYTON_DAISY_BOARD.value
        )
        params = (
            BrainFlowInputParams()
            if self.sim
            else BrainFlowInputParams(serial_port=SERIAL_PORT)
        )
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

    def get_current_data(self, n_samples=100):
        return self.board.get_current_board_data(n_samples)

    def save_and_exit(self):
        session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()
        DataFilter.write_file(data, f"Session_{session_time}.csv", "w")


class DemoHandler:
    def get_current_data(self):
        return randint(0, 2)


if __name__ == "__main__":
    handler = OpenBCIHandler(sim=True)
    sleep(1)
    handler.save_and_exit()
