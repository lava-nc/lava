import numpy as np
import unittest
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.core.model.py.ports import PyPortMessage
from lava.magma.core.model.interfaces import PortMessageFormat


class MockInterface:
    def __init__(self, smm):
        self.smm = smm


def get_channel(smm, msg, size, name="test_channel") -> PyPyChannel:
    mock = MockInterface(smm)
    channel = PyPyChannel(
        message_infrastructure=mock,
        src_name=name,
        dst_name=name,
        shape=msg.shape,
        dtype=msg.dtype,
        size=size+3
    )
    return channel


class TestPyPyChannelSingleProcess(unittest.TestCase):
    def test_send_recv_single_process(self):
        smm = SharedMemoryManager()
        try:
            smm.start()

            data = np.ones((2, 2, 2))
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                data.size,
                data
            )
            channel = get_channel(smm, message.payload, message.num_elements)

            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(message=message)
            result = channel.dst_port.recv()
            assert np.array_equal(result[2], data)
        finally:
            smm.shutdown()

    def test_send_recv_single_process_2d_data(self):
        smm = SharedMemoryManager()
        try:
            smm.start()

            data = np.random.randint(100, size=(100, 100), dtype=np.int32)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                data.size,
                data
            )

            channel = get_channel(smm, message.payload, message.num_elements)

            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(message=message)
            result = channel.dst_port.recv()
            assert np.array_equal(result[2], data)
        finally:
            smm.shutdown()

    def test_send_recv_single_process_1d_data(self):
        smm = SharedMemoryManager()
        try:
            smm.start()

            data = np.random.randint(1000, size=100, dtype=np.int16)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                data.size,
                data
            )
            channel = get_channel(smm, message.payload, message.num_elements)

            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(message=message)
            result = channel.dst_port.recv()
            assert np.array_equal(result[2], data)
        finally:
            smm.shutdown()


class DummyProcess(Process):
    """Wrapper around multiprocessing.Process to start channels"""

    def __init__(self, ports=None, **kwargs):
        super().__init__(**kwargs)
        self._ports = ports

    def run(self):
        for c in self._ports:
            c.start()
        super().run()


def source(shape, port):
    msg = np.zeros(shape=shape)
    for i in range(2):
        for j in range(2):
            msg[i][j] = 1
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                msg.size,
                msg
            )
            port.send(message)


def sink(shape, port):
    expected_result = np.array(
        [[[2, 0], [0, 0]], [[1, 2], [0, 0]], [[1, 1], [2, 0]], [[1, 1], [1, 2]]]
    )
    for i in range(4):
        msg = port.recv()[2]
        assert msg.shape == shape
        assert np.array_equal(
            msg, expected_result[i]
        ), f"Mismatch {msg=} {expected_result[i]=}"


def buffer(shape, recv_port, send_port):
    for i in range(2):
        for j in range(2):
            peek = recv_port.peek()[2]
            msg = recv_port.recv()[2]
            assert np.array_equal(peek, msg)
            assert msg.shape == shape
            msg[i][j] += 1
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                msg.size,
                msg
            )
            send_port.send(message)


class TestPyPyChannelMultiProcess(unittest.TestCase):
    def test_send_recv_relay(self):
        smm = SharedMemoryManager()
        try:
            smm.start()
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                4,
                np.ones((2, 2))
            )

            channel_source_to_buffer = get_channel(
                smm,
                message.payload,
                message.payload.size,
                name="channel_source_to_buffer"
            )
            channel_buffer_to_sink = get_channel(
                smm,
                message.payload,
                message.payload.size,
                name="channel_buffer_to_sink"
            )

            jobs = [
                DummyProcess(
                    ports=(channel_source_to_buffer.src_port,),
                    target=source,
                    args=(
                        message.data.shape,
                        channel_source_to_buffer.src_port,
                    ),
                ),
                DummyProcess(
                    ports=(
                        channel_source_to_buffer.dst_port,
                        channel_buffer_to_sink.src_port,
                    ),
                    target=buffer,
                    args=(
                        message.data.shape,
                        channel_source_to_buffer.dst_port,
                        channel_buffer_to_sink.src_port,
                    ),
                ),
                DummyProcess(
                    ports=(channel_buffer_to_sink.dst_port,),
                    target=sink,
                    args=(
                        message.data.shape,
                        channel_buffer_to_sink.dst_port,
                    ),
                ),
            ]
            for p in jobs:
                p.start()
            for p in jobs:
                p.join()
        finally:
            smm.shutdown()


if __name__ == "__main__":
    unittest.main()
