import zmq
from random import randint
from numpy.random import normal
from data_api.data_handler import DemoHandler, OpenBCIHandler
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
            message = self.socket.recv()
            raw = self.data_handler.get_current_data(n_samples=100)
            processed = self.extractor.process(raw)
            prediction = self.predictor.predict(processed)
            print(f"raw: {raw}, processed: {processed}, pred: {prediction}")
            command = self.commands[prediction]
            self.socket.send(command)
            print(f"Sent command: {command}")


if __name__ == "__main__":
    server = BCIServer(sim=True)
    server.run()
