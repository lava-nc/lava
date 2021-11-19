# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from lava.magma.compiler.channels.interfaces import AbstractCspSendPort, \
    AbstractCspRecvPort

from lava.magma.core.model.py.ports import PyInPort, PyInPortVectorDense, \
    PyOutPort, PyOutPortVectorDense

from lava.magma.compiler.channels.pypychannel import PyPyChannel


class MockInterface:
    def __init__(self, smm):
        self.smm = smm


def get_channel(smm, data, size, name="test_channel") -> PyPyChannel:
    mock = MockInterface(smm)
    return PyPyChannel(
        message_infrastructure=mock,
        src_name=name,
        dst_name=name,
        shape=data.shape,
        dtype=data.dtype,
        size=size
    )


class TestPyPorts(unittest.TestCase):
    def test_py_in_port_vector_dense_probe_single_csp_recv_port(self):
        """Tests PyInPortVectorDense probe method for single recv_csp_port"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            data = np.random.randint(5, size=(2, 4))

            channel = get_channel(smm, data, data.size)
            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorDense([send_csp_port], None)
            recv_py_port: PyInPort = \
                PyInPortVectorDense([recv_csp_port], None)

            recv_py_port.start()
            send_py_port.start()

            # Send data through PyOutPort
            send_py_port.send(data)
            # Sleep to let message reach the PyInPort
            time.sleep(0.001)
            # Probe PyInPort
            probe_value = recv_py_port.probe()

            # probe_value should be True if message reached the PyInPort
            self.assertTrue(probe_value)

            # Get data that reached PyInPort to empty buffer
            _ = recv_py_port.recv()
            # Probe PyInPort
            probe_value = recv_py_port.probe()

            # probe_value should be False since PyInPort's buffer was emptied
            self.assertFalse(probe_value)
        finally:
            smm.shutdown()

    def test_py_in_port_vector_dense_probe_multi_csp_recv_port(self):
        """Tests PyInPortVectorDense probe method for multiple recv_csp_port"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            data = np.random.randint(5, size=(2, 4))

            channel_1 = get_channel(smm, data, data.size)
            send_csp_port_1: AbstractCspSendPort = channel_1.src_port
            recv_csp_port_1: AbstractCspRecvPort = channel_1.dst_port

            channel_2 = get_channel(smm, data, data.size)
            send_csp_port_2: AbstractCspSendPort = channel_2.src_port
            recv_csp_port_2: AbstractCspRecvPort = channel_2.dst_port

            send_py_port_1: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_1], None)
            send_py_port_2: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_2], None)
            recv_py_port: PyInPort = PyInPortVectorDense(
                [recv_csp_port_1, recv_csp_port_2], None)

            recv_py_port.start()
            send_py_port_1.start()
            send_py_port_2.start()

            # Send data through first PyOutPort
            send_py_port_1.send(data)
            # Send data through second PyOutPort
            send_py_port_2.send(data)
            # Sleep to let message reach the PyInPort
            time.sleep(0.001)
            # Probe PyInPort
            probe_value = recv_py_port.probe()

            # probe_value should be True if message reached the PyInPort
            self.assertTrue(probe_value)

            # Get data that reached PyInPort to empty buffer
            _ = recv_py_port.recv()
            # Probe PyInPort
            probe_value = recv_py_port.probe()

            # probe_value should be False since PyInPort's buffer was emptied
            self.assertFalse(probe_value)
        finally:
            smm.shutdown()


if __name__ == '__main__':
    unittest.main()
