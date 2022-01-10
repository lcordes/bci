import zmq


class BCIClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        # Implement server heartbeat
        # self.socket.send(b"Command")
        # command = self.socket.recv().decode("UTF-8")
        # print(command)
        print("Connected to BCI server")

    def request_command(self):
        self.socket.send(b"Command")

    def get_command(self):
        return self.socket.recv().decode("UTF-8")
