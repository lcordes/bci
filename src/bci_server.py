import argparse
import zmq
from data_api.data_handler import OpenBCIHandler
from feature_extraction.preprocessing import DemoExtractor
from classifier.classifier import DemoPredictor


class BCIServer:
    def __init__(self, sim):
        self.sim = sim
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("BCI server started")

        self.data_handler = OpenBCIHandler(sim=self.sim)
        self.extractor = DemoExtractor()
        self.predictor = DemoPredictor("model_file.csv")

        self.commands = [b"left", b"right", b"up"]
        self.running = True

    def run(self):
        while self.running:
            if self.data_handler.status == "no_connection":
                print("\n", "No headset connection, use '--sim' for simulated data")
                break

            message = self.socket.recv()
            raw = self.data_handler.get_current_data(n_samples=100)
            processed = self.extractor.process(raw)
            prediction = self.predictor.predict(processed)
            print(f"processed: {processed}, pred: {prediction}")
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
