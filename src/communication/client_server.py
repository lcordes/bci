import zmq


class Client:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        print("Connected to BCI server")

    def request_command(self):
        self.socket.send(b"Command")

    def get_command(self):
        return self.socket.recv().decode("UTF-8")


class Server:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    def await_request(self):
        _ = self.socket.recv()

    def send_response(self, string):
        self.socket.send(string.encode("utf-8"))
