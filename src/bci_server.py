import argparse
import zmq
import numpy as np
from time import time, sleep
from data_acquisition.data_handler import OpenBCIHandler
from feature_extraction.extractors import Extractor
from classification.classifiers import Classifier


class BCIServer:
    def __init__(self, sim, model_name):
        self.sim = sim
        self.models_path = "data/models/"  # add path from parent directory in front
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("BCI server started")

        self.data_handler = OpenBCIHandler(sim=self.sim)
        self.sampling_rate = self.data_handler.get_sampling_rate()
        self.extractor = Extractor(type="CSP")
        self.extractor.load_model(model_name)
        self.predictor = Classifier(type="LDA")
        self.predictor.load_model(model_name)
        self.commands = {"1": b"left", "2": b"right", "3": b"up"}
        self.running = True

    def run(self):
        while self.running:
            if self.data_handler.status == "no_connection":
                print("\n", "No headset connection, use '--sim' for simulated data")
                break

            _ = self.socket.recv()
            start = time()
            raw = self.data_handler.get_current_data()
            print("Raw shape:", raw.shape)
            features = self.extractor.transform(raw)
            print("Features shape:", features.shape)
            prediction = int(self.predictor.predict(features))
            print(f"Prediction: {prediction}")
            command = self.commands[str(prediction)]
            self.socket.send(command)
            print(f"Sent command: {command}")
            delta = np.round(time() - start, 3)
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
        "--model",
        nargs="?",
        default="model1",
        const="model1",
        help="Specifies the Extractor and Classifier model name.",
    )
    args = parser.parse_args()

    server = BCIServer(sim=args.sim, model_name=args.model)
    server.run()
