import argparse
import numpy as np
from time import time, sleep
from communication.pub_sub import Publisher
from communication.client_server import Server
from data_acquisition.data_handler import OpenBCIHandler
from feature_extraction.extractors import MIExtractor, ConcentrationExtractor
from classification.classifiers import Classifier
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]

TIMESTEP = 2  # in s


class BCIServer:
    def __init__(self, board_type, concentration, eval, model_name):
        self.board__type = board_type
        self.eval = eval
        self.models_path = (
            f"{DATA_PATH}/models/"  # add path from parent directory in front
        )
        self.concentration = concentration

        if self.concentration:
            self.extractor = ConcentrationExtractor(
                sr=self.sampling_rate, board_id=self.board_id
            )
        else:

            self.extractor = MIExtractor(type="CSP")
            self.extractor.load_model(model_name)
            self.n_channels = self.extractor.get_n_channels()
            self.predictor = Classifier(type="LDA")
            self.predictor.load_model(model_name)
            self.commands = {"1": b"left", "2": b"right", "3": b"up"}

        self.data_handler = OpenBCIHandler(board_type=board_type)
        self.sampling_rate = self.data_handler.get_sampling_rate()
        self.board_id = self.data_handler.get_board_id()

        if self.eval:
            self.server = Server()
        else:
            self.publisher = Publisher()

        self.running = True
        print("BCI server started")

    def get_mi(self):
        raw = self.data_handler.get_current_data(self.n_channels)
        features = self.extractor.transform(raw)
        probs = self.predictor.predict_probs(features)
        probs = [np.round(prob, 3) for prob in probs]

        prediction = int(self.predictor.predict(features))
        print("Raw shape:", raw.shape)
        print("Features shape:", features.shape)
        print(f"Class probabilities: {probs}")
        print(f"Prediction: {prediction}")

        return probs, prediction

    def get_concentration(self):
        raw = self.data_handler.get_concentration_data()
        concentration = self.extractor.get_concentration(raw)
        print(f"Sent concentration: {concentration}")
        return concentration

    def run(self):
        while self.running:
            if self.eval:
                self.server.await_request()
            else:
                sleep(TIMESTEP)
            if self.data_handler.status == "no_connection":
                print(
                    "\n",
                    "No headset connection, use '--board synthetic' for simulated data",
                )
                break

            start = time()
            if self.concentration:
                event = self.get_concentration()
                topic = "Concentration"
            else:
                probs, event = self.get_mi()
                topic = "MI"

            if self.eval:
                self.server.send_response(f"{probs[0]};{probs[1]};{probs[2]}")
            else:
                if not event == "No event":
                    self.publisher.send_event(event, topic)

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
        "--eval",
        help="Use server for experiment evaluation",
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
        help="Specify the Extractor and Classifier model name.",
    )

    parser.add_argument(
        "--board",
        nargs="?",
        default="daisy",
        const="daisy",
        help="Use synthetic or cyton board instead of daisy",
    )
    args = parser.parse_args()
    print(args.board)
    server = BCIServer(
        sim=args.sim,
        concentration=args.concentration,
        eval=args.eval,
        model_name=args.model,
        board_type=args.type,
    )
    server.run()
