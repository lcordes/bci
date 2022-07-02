import argparse
import numpy as np
from time import time, sleep
from communication.pub_sub import Publisher
from communication.client_server import Server
from data_acquisition.data_handler import OpenBCIHandler
from feature_extraction.extractors import CSPExtractor
from classification.classifiers import LDAClassifier
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
TIMESTEP = 2  # in s


class BCIServer:
    def __init__(self, board_type, client_model, model_name):
        self.client_model = client_model
        self.models_path = (
            f"{DATA_PATH}/models/"  # add path from parent directory in front
        )
        self.extractor = CSPExtractor()
        self.extractor.load_model(model_name)
        self.n_channels = 8  # TODO update this to load from model_info
        self.predictor = LDAClassifier()
        self.predictor.load_model(model_name)
        self.data_handler = OpenBCIHandler(board_type=board_type)
        self.sampling_rate = self.data_handler.get_sampling_rate()

        if self.client_model:
            self.server = Server()
        else:
            self.publisher = Publisher()

        self.running = True
        print(f"BCI server started, using '{model_name}' for prediction")

    def get_prediction(self):
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

    def run(self):
        while self.running:
            if self.client_model:
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
            probs, event = self.get_prediction()
            if self.client_model:
                self.server.send_response(f"{probs[0]};{probs[1]};{probs[2]}")
            else:
                if not event == "No event":
                    self.publisher.send_event(event, "MI")
            delta = np.round(time() - start, 3)
            print(f"Processing time: {delta} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        help="Use client-server instead of publisher-subscriber pattern.",
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
        choices=["synthetic", "cyton", "daisy"],
        help="Use synthetic or cyton board instead of daisy",
    )
    args = parser.parse_args()
    server = BCIServer(
        client_model=args.client,
        model_name=args.model,
        board_type=args.board,
    )
    server.run()
