from MessageInfrastructurePywrapper import SendPort as CppSendPort
from MessageInfrastructurePywrapper import Channel as CppChannel
import numpy as np


class SendPort:
    def __init__(self, send_port):
        self._cpp_send_port = send_port

    def send(self, data):
        # Use np.copy to handle slices input
        self._cpp_send_port.send(np.copy(data))

    def start(self):
        self._cpp_send_port.start()

    def probe(self):
        return self._cpp_send_port.probe()

    def join(self):
        self._cpp_send_port.join()

    @property
    def name(self):
        return self._cpp_send_port.name

    def size(self):
        return self._cpp_send_port.size()

    def get_channel_type(self):
        return self._cpp_send_port.get_channel_type()


class Channel(CppChannel):

    @property
    def src_port(self):
        return SendPort(super().src_port)
