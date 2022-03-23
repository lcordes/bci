import argparse
import zmq
import numpy as np
from time import time, sleep
from apps.concentration import TIMESTEP
from communication.pub_sub import Publisher
from data_acquisition.data_handler import OpenBCIHandler
from feature_extraction.extractors import MIExtractor, ConcentrationExtractor
from classification.classifiers import Classifier

TIMESTEP = 0.1


class BCIServer:
    def __init__(self, sim, concentration, silent, model_name):
        self.sim = sim
        self.silent = silent
        self.models_path = "data/models/"  # add path from parent directory in front
        self.concentration = concentration
        self.data_handler = OpenBCIHandler(sim=self.sim)
        self.sampling_rate = self.data_handler.get_sampling_rate()
        self.board_id = self.data_handler.get_board_id()

        if self.concentration:
            self.extractor = ConcentrationExtractor(
                sr=self.sampling_rate, board_id=self.board_id
            )
            self.publisher = Publisher("Concentration")

        else:

            self.extractor = MIExtractor(type="CSP")
            self.extractor.load_model(model_name)
            self.predictor = Classifier(type="LDA")
            self.predictor.load_model(model_name)
            self.commands = {"1": b"left", "2": b"right", "3": b"up"}
            self.publisher = Publisher("MI")

        self.running = True
        print("BCI server started")

    def get_mi(self):
        raw = self.data_handler.get_current_data()
        features = self.extractor.transform(raw)
        prediction = int(self.predictor.predict(features))
        if not self.silent:
            print("Raw shape:", raw.shape)
            print("Features shape:", features.shape)
            print(f"Prediction: {prediction}")
        return prediction

    def get_concentration(self):
        raw = self.data_handler.get_concentration_data()
        concentration = self.extractor.get_concentration(raw)
        if not self.silent:
            print(f"Sent concentration: {concentration}")
        return concentration

    def run(self):
        while self.running:
            sleep(TIMESTEP)
            if self.data_handler.status == "no_connection":
                print("\n", "No headset connection, use '--sim' for simulated data")
                break

            start = time()
            if self.concentration:
                event = self.get_concentration()
            else:
                event = self.get_mi()

            if not event == "No event":
                self.publisher.send_event(event)

            delta = np.round(time() - start, 3)
            if not self.silent:
                print(f"Processing time: {delta} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        help="use simulated data instead of headset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--concentration",
        help="Estimate concentration instead of MI classification",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--model",
        nargs="?",
        default="model1",
        const="model1",
        help="Specifies the Extractor and Classifier model name.",
    )
    parser.add_argument(
        "--silent",
        help="Do not print message info to the console",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    server = BCIServer(
        sim=args.sim,
        concentration=args.concentration,
        silent=args.silent,
        model_name=args.model,
    )
    server.run()
