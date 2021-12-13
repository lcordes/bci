import zmq
from random import choice


class BCIServer:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("BCI server started")
        self.running = True

    def run(self):
        while self.running:
            message = self.socket.recv()
            command = choice([b"left", b"right", b"up", b"down"])
            self.socket.send(command)
            print(f"Sent command: {command}")


if __name__ == "__main__":
    server = BCIServer()
    server.run()
