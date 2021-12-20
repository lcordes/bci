import zmq
from random import randint
from numpy.random import normal
from data_api.data_handler import DemoHandler, OpenBCIHandler
from feature_extraction.preprocessing import DemoExtractor
from classifier.classifier import DemoPredictor


class BCIServer:
    def __init__(self, mode):
        self.mode = mode
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("BCI server started")

        if self.mode == "demo":
            self.data_handler = DemoHandler()
            self.extractor = DemoExtractor()
            self.predictor = DemoPredictor("model_file.csv")
        else:
            self.data_handler = OpenBCIHandler(n_samples=100)

        self.commands = [b"left", b"right", b"up"]
        self.running = True

    def run(self):
        while self.running:
            message = self.socket.recv()
            raw = self.data_handler.get_current_data()
            processed = self.extractor.process(raw)
            prediction = self.predictor.predict(processed)
            print(f"raw: {raw}, processed: {processed}, pred: {prediction}")
            command = self.commands[prediction]
            self.socket.send(command)
            print(f"Sent command: {command}")


if __name__ == "__main__":
    server = BCIServer(mode="demo")
    server.run()
