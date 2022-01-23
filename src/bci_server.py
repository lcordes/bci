import argparse
import zmq
from data_acquisition.data_handler import OpenBCIHandler
from feature_extraction.preprocessing import CspExtractor, DemoExtractor
from classification.linear_discriminant_analysis import LDA


class BCIServer:
    def __init__(self, sim):
        self.sim = sim
        self.models_path = "data/models/"  # add path from parent directory in front
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("BCI server started")

        self.data_handler = OpenBCIHandler(sim=self.sim)
        self.sampling_rate = self.data_handler.get_sampling_rate()
        self.extractor = DemoExtractor()
        self.extractor = CspExtractor(
            sampling_rate=self.sampling_rate, models_path=self.models_path
        )
        self.predictor = LDA()

        self.commands = [b"left", b"right", b"up"]
        self.running = True

    def run(self):
        while self.running:
            if self.data_handler.status == "no_connection":
                print("\n", "No headset connection, use '--sim' for simulated data")
                break

            message = self.socket.recv()
            raw = self.data_handler.get_current_data()
            features = self.extractor.test(raw)
            prediction = self.predictor.predict(features)
            print(f"Prediction: {prediction}")
            command = self.commands[prediction]
            self.socket.send(command)
            print(f"Sent command: {command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim",
        help="use simulated data instead of headset",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    server = BCIServer(sim=args.sim)
    server.run()
