import argparse
import numpy as np
from time import time
from communication.client_server import Server
from data_acquisition.data_handler import OpenBCIHandler, RecordingHandler
from data_acquisition.preprocessing import preprocess_trial
from classification.train_test_model import load_model


class BCIServer:
    def __init__(self, board_type, model_name, recording):
        self.recording = recording
        if self.recording:
            model_name = f"LDA_optimal_{self.recording}"
        self.extractor, self.predictor = load_model(model_name)
        self.config = self.predictor.model.config
        self.labels = {1: "left", 2: "right", 3: "down"}

        if self.recording:
            self.data_handler = RecordingHandler(
                recording_name=f"{self.recording}",
                config=self.config,
            )
        else:
            self.data_handler = OpenBCIHandler(board_type=board_type)
        self.sampling_rate = self.data_handler.get_sampling_rate()
        self.server = Server()
        self.running = True
        print(f"BCI server started, using '{model_name}' for prediction")

    def get_prediction(self, command):
        if self.recording:
            raw = self.data_handler.get_current_data(label=command)
        else:
            raw = self.data_handler.get_current_data(self.config["n_channels"])
        processed = preprocess_trial(raw, self.sampling_rate, self.config)
        features = self.extractor.transform(processed)
        probs = self.predictor.predict_probs(features)
        probs = [np.round(prob, 3) for prob in probs]

        prediction = self.labels[int(self.predictor.predict(features))]

        print("Raw shape:", raw.shape)
        print("Processed shape:", processed.shape)
        print("Features shape:", features.shape)
        print(f"Class probabilities: {probs}")
        print(f"Prediction: {prediction}")
        print(f"Label: {command}")

        return probs, prediction

    def run(self):
        while self.running:
            command = self.server.await_request()
            if not self.recording and self.data_handler.status == "no_connection":
                print(
                    "No headset connection, use '--board synthetic' for simulated data"
                )
                break

            start = time()
            probs, event = self.get_prediction(command)
            self.server.send_response(f"{event};{probs[0]};{probs[1]};{probs[2]}")
            delta = np.round(time() - start, 3)
            print(f"Processing time: {delta} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="?",
        default="model",
        const="model",
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
    parser.add_argument(
        "--recording",
        nargs="?",
        default=None,
        const=None,
        help="Simulate live data from a given user recording",
    )
    args = parser.parse_args()
    server = BCIServer(
        model_name=args.model,
        board_type=args.board,
        recording=args.recording,
    )
    server.run()
