import zmq
from time import sleep


class Client:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self.status = self.server_probe()

    def request_command(self):
        self.socket.send(b"Command")

    def get_command(self, no_block=0):
        return self.socket.recv(no_block).decode("UTF-8")

    def server_probe(self):
        self.request_command()
        sleep(0.1)
        try:
            self.get_command(no_block=1)
            return "connected"
        except Exception:
            return "no connection"

    def connected(self):
        return True if self.status == "connected" else False


class Server:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    def await_request(self):
        _ = self.socket.recv()

    def send_response(self, string):
        self.socket.send(string.encode("utf-8"))
