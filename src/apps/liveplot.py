import argparse
import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

pg.setConfigOption("background", "w")
LINE_COLOUR = "#081abf"

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_api.data_handler import OpenBCIHandler


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        print(type(self.board_id))
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Live EEG Data", size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis("left", False)
            p.setMenuEnabled("left", False)
            p.showAxis("bottom", False)
            p.setMenuEnabled("bottom", False)
            self.plots.append(p)
            curve = p.plot(pen={"color": LINE_COLOUR})
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                data[channel],
                self.sampling_rate,
                51.0,
                100.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )
            DataFilter.perform_bandpass(
                data[channel],
                self.sampling_rate,
                51.0,
                100.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )
            DataFilter.perform_bandstop(
                data[channel],
                self.sampling_rate,
                50.0,
                4.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )
            DataFilter.perform_bandstop(
                data[channel],
                self.sampling_rate,
                60.0,
                4.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        help="use simulated data instead of headset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--log",
        help="write session data to csv",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    data_handler = OpenBCIHandler(sim=args.sim)
    board_shim = data_handler.board

    if data_handler.status == "no_connection":
        print("\n", "No headset connection, use '--sim' for simulated data")
    else:
        try:
            Graph(board_shim)

        except BaseException:
            logging.warning("Exception", exc_info=True)
        finally:
            logging.info("End")
            if board_shim.is_prepared():
                logging.info("Releasing session")
                if args.log:
                    data_handler.save_and_exit()
                else:
                    board_shim.release_session()
