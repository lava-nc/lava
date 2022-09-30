import time
import numpy as np
import unittest
from multiprocessing import Process

from message_infrastructure import (
    ChannelBackend,
    Channel,
    AbstractTransferPort,
    ChannelQueueSize
)


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def get_channel(data, name="test_channel") -> Channel:
    return Channel(
        ChannelBackend.SHMEMCHANNEL,
        ChannelQueueSize,
        nbytes_cal(data.shape, data.dtype),
        name + "src",
        name + "dst")


class TestPyPyChannelSingleProcess(unittest.TestCase):
    def test_send_recv_single_process(self):
        data = np.ones((2, 2, 2))
        channel = get_channel(data)
        try:
            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(data)
            result = channel.dst_port.recv()
            assert np.array_equal(result, data)
        finally:
            channel.src_port.join()
            channel.dst_port.join()

    def test_send_recv_single_process_2d_data(self):
        data = np.random.randint(100, size=(100, 100), dtype=np.int32)
        channel = get_channel(data)
        try:
            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(data)
            result = channel.dst_port.recv()
            assert np.array_equal(result, data)
        finally:
            channel.src_port.join()
            channel.dst_port.join()

    def test_send_recv_single_process_1d_data(self):
        data = np.random.randint(1000, size=100, dtype=np.int16)
        channel = get_channel(data)
        try:
            channel.src_port.start()
            channel.dst_port.start()

            channel.src_port.send(data)
            result = channel.dst_port.recv()
            assert np.array_equal(result, data)
        finally:
            channel.src_port.join()
            channel.dst_port.join()


class DummyProcess(Process):
    """Wrapper around multiprocessing.Process to start channels"""

    def __init__(self, ports=None, **kwargs):
        super().__init__(**kwargs)
        self._ports = ports

    def run(self):
        for c in self._ports:
            c.start()
        # need to wait all port started.
        time.sleep(0.01)
        super().run()
        for c in self._ports:
            c.join()


def source(shape, port):
    msg = np.zeros(shape=shape)
    for i in range(2):
        for j in range(2):
            msg[i][j] = 1
            port.send(msg)


def sink(shape, port):
    expected_result = np.array(
        [[[2, 0], [0, 0]], [[1, 2], [0, 0]], [[1, 1], [2, 0]], [[1, 1], [1, 2]]]
    )
    for i in range(4):
        msg = port.recv()
        assert msg.shape == shape
        assert np.array_equal(
            msg, expected_result[i]
        ), f"Mismatch {msg=} {expected_result[i]=}"


def buffer(shape, dst_port, src_port):
    for i in range(2):
        for j in range(2):
            peek = dst_port.peek()
            msg = dst_port.recv()
            assert np.array_equal(peek, msg)
            assert msg.shape == shape
            msg[i][j] += 1
            src_port.send(msg)


class TestPyPyChannelMultiProcess(unittest.TestCase):
    def test_send_recv_relay(self):
        data = np.ones((2, 2))
        channel_source_to_buffer = get_channel(
            data, name="channel_source_to_buffer"
        )
        channel_buffer_to_sink = get_channel(
            data, name="channel_buffer_to_sink"
        )
        jobs = [
            DummyProcess(
                ports=(channel_source_to_buffer.src_port,),
                target=source,
                args=(
                    data.shape,
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
                    data.shape,
                    channel_source_to_buffer.dst_port,
                    channel_buffer_to_sink.src_port,
                ),
            ),
            DummyProcess(
                ports=(channel_buffer_to_sink.dst_port,),
                target=sink,
                args=(
                    data.shape,
                    channel_buffer_to_sink.dst_port,
                ),
            ),
        ]
        for p in jobs:
            p.start()
        for p in jobs:
            p.join()


if __name__ == "__main__":
    unittest.main()
