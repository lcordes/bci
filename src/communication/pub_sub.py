import time
import zmq

HOST = "127.0.0.1"
PORT = "5001"


class Publisher:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{HOST}:{PORT}")
        time.sleep(1)  # Check if this is needed (for bind to execute)

    def send_event(self, event, topic):
        self.socket.send_string(f"{topic} {event}")


class Subscriber:
    def __init__(self, topic):
        self.topic = topic
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{HOST}:{PORT}")
        self.socket.subscribe(topic)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        time.sleep(1)  # Check if this is needed

    def get_event(self):
        evts = dict(self.poller.poll(timeout=10))
        event = "No event"
        if self.socket in evts:
            topic, event = self.socket.recv_string().split()
        return event
