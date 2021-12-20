from random import randint
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BoardShim.enable_dev_board_logger()
SERIAL_PORT = ""


class OpenBCIHandler:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        params = BrainFlowInputParams(serial_port=SERIAL_PORT)
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        self.board.prepare_session()
        self.board.start_stream()

    def get_current_data(self):
        return self.board.get_current_board_data(self.n_samples)

    def save_and_exit(self):
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()
        print(data)  # write data to file here


class DemoHandler:
    def get_current_data(self):
        return randint(0, 2)
