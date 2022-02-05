# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import re
import scipy.sparse as sparse
from scipy.sparse.csr import csr_matrix


from lava.magma.compiler.channels.interfaces import AbstractCspSendPort, \
    AbstractCspRecvPort
from lava.magma.compiler.channels.pypychannel import PyPyChannel
import lava.magma.core.model.py.ports as ports
from lava.magma.core.model.interfaces import (
    PortMessageFormat
)
from lava.magma.core.model.py.ports import (
    PyInPort,
    PyOutPort,
    PyInPortVectorDense,
    PyOutPortVectorDense,
    PyPortMessage
)


class MockInterface:
    def __init__(self, smm):
        self.smm = smm


class PortCombos:
    def __init__(
        self
    ):
        self.port_combos = [('VectorDense', 'VectorDense')]
        # noqa E131 ('VectorDense', 'VectorSparse'),
        # ('VectorDense', 'ScalarDense'),
        # ('VectorDense', 'ScalarSparse'),
        # ('VectorSparse', 'VectorSparse'),
        # ('VectorSparse', 'VectorDense'),
        # ('VectorSparse', 'ScalarDense'),
        # ('VectorSparse', 'ScalarSparse'),
        # ('ScalarDense', 'ScalarDense'),
        # ('ScalarDense', 'ScalarSparse'),
        # ('ScalarDense', 'VectorDense'),
        # ('ScalarDense', 'VectorSparse'),
        # ('ScalarSparse', 'ScalarSparse'),
        # ('ScalarSparse', 'ScalarDense'),
        # ('ScalarSparse', 'VectorDense'),
        # ('ScalarSparse', 'VectorSparse')],
        self.data = {
            "VectorDense": np.ones((2, 4)),
            "VectorSparse": sparse.random(
                5, 5, density=0.05, random_state=1, format='csr'),
            "ScalarDense": np.ones((1, 4)),
            "ScalarSparse": sparse.random(
                5, 0, density=0.05, random_state=1, format='csr')
        }


def get_channel(smm, msg, size, name="test_channel") -> PyPyChannel:
    mock = MockInterface(smm)
    return PyPyChannel(
        message_infrastructure=mock,
        src_name=name,
        dst_name=name,
        shape=msg.payload.shape,
        dtype=msg.payload.dtype,
        size=size
    )


def get_ports(smm, msg, out_port_name, in_port_name):
    channel = get_channel(smm,
                          msg,
                          size=msg.num_elements)

    send_csp_port: AbstractCspSendPort = channel.src_port
    recv_csp_port: AbstractCspRecvPort = channel.dst_port

    send_py_port: PyOutPort = \
        getattr(ports, 'PyOutPort'
                + str(out_port_name))(process_model=None,
                                      csp_send_ports=send_csp_port,
                                      shape=msg.data.shape,
                                      d_type=msg.data.dtype)
    recv_py_port: PyInPort = \
        getattr(ports, 'PyInPort'
                + str(in_port_name))(process_model=None,
                                     csp_recv_ports=recv_csp_port,
                                     shape=msg.data.shape,

                                     d_type=msg.data.dtype)
    return send_py_port, recv_py_port


class TestPyPorts(unittest.TestCase):
    def test_pyports(self):
        """Tests sending data over combination of Vector/Scalar Dense/Sparse
        port combinations"""
        port_combos = PortCombos()

        for port_combo in port_combos.port_combos:
            smm = SharedMemoryManager()
            try:
                smm.start()

                data = port_combos.data[str(port_combo[0])]
                # Add underscore between portcombo to get type
                send_port_type = re.sub(
                    r"(\w)([A-Z])", r"\1_\2", port_combo[0]
                ).upper()
                msg = PyPortMessage(
                    PortMessageFormat[send_port_type],
                    data.size,
                    data
                )

                send_py_port, recv_py_port = get_ports(
                    smm,
                    msg,
                    port_combo[0],
                    port_combo[1]
                )

                recv_py_port.start()
                send_py_port.start()

                if ('Sparse' in port_combo[0]):
                    send_py_port.send(data.toarray(), data.indices)
                else:
                    send_py_port.send(data)

                if ('Sparse' in port_combo[1]):
                    recv_data, recv_idx = recv_py_port.recv()
                    data = csr_matrix(data)

                    # ## Debug - Remove before merge
                    print("")
                    print("If working, below arrays are np.array_equal")
                    print(port_combo)
                    print("")
                    print(data.toarray())
                    print("")
                    print(data.indices)
                    print(" equals: ")
                    print(recv_data)
                    print("")
                    print(recv_idx)
                    # ##

                    assert np.array_equal(data.toarray(), recv_data)
                    assert np.array_equal(data.indices, recv_idx)
                else:
                    recv_data = recv_py_port.recv()

                    # ## Debug - Remove before merge
                    print("")
                    print("If working, below arrays are np.array_equal")
                    print(port_combo)
                    print("")
                    print(data)
                    print(" equals: ")
                    print(recv_data)
                    # ##

                    assert np.array_equal(data, recv_data)
            finally:
                smm.shutdown()

    def probe_test_routine(self, cls):
        """Routine that tests probe method on one implementation of PyInPorts.
        """
        smm = SharedMemoryManager()

        try:
            smm.start()

            data = np.ones((4, 4))
            msg = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                data.size,
                data
            )

            channel_1 = get_channel(smm, msg, msg.num_elements)
            send_csp_port_1: AbstractCspSendPort = channel_1.src_port
            recv_csp_port_1: AbstractCspRecvPort = channel_1.dst_port

            channel_2 = get_channel(smm, msg, msg.num_elements)
            send_csp_port_2: AbstractCspSendPort = channel_2.src_port
            recv_csp_port_2: AbstractCspRecvPort = channel_2.dst_port

            # Create two different PyOutPort
            send_py_port_1: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_1], None)
            send_py_port_2: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_2], None)
            # Create PyInPort with current implementation
            recv_py_port: PyInPort = \
                cls([recv_csp_port_1, recv_csp_port_2], None)

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

    def test_py_in_port_probe(self):
        """Tests PyInPort probe method on all implementations of PyInPorts."""
        # TODO: (GK) Add other classes when we support them
        classes = [PyInPortVectorDense]

        for cls in classes:
            self.probe_test_routine(cls)


if __name__ == '__main__':
    unittest.main()
