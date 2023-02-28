# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import time
import numpy as np
import typing as ty
import functools as ft

from lava.magma.compiler.channels.interfaces import (
    AbstractCspPort,
    AbstractCspSendPort,
    AbstractCspRecvPort)
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.core.model.py.ports import (
    PyInPort,
    PyInPortVectorDense,
    PyOutPort,
    PyOutPortVectorDense,
    VirtualPortTransformer,
    IdentityTransformer)
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager
)


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
    def probe_test_routine(self, cls):
        """Routine that tests probe method on one implementation of PyInPorts.
        """
        smm = SharedMemoryManager()

        try:
            smm.start()

            data = np.ones((4, 4))

            channel_1 = get_channel(smm, data, data.size)
            send_csp_port_1: AbstractCspSendPort = channel_1.src_port
            recv_csp_port_1: AbstractCspRecvPort = channel_1.dst_port

            channel_2 = get_channel(smm, data, data.size)
            send_csp_port_2: AbstractCspSendPort = channel_2.src_port
            recv_csp_port_2: AbstractCspRecvPort = channel_2.dst_port

            # Create two different PyOutPort
            send_py_port_1: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_1], None, data.shape,
                                     data.dtype)
            send_py_port_2: PyOutPort = \
                PyOutPortVectorDense([send_csp_port_2], None, data.shape,
                                     data.dtype)
            # Create PyInPort with current implementation
            recv_py_port: PyInPort = \
                cls([recv_csp_port_1, recv_csp_port_2], None, data.shape,
                    data.dtype)

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
        classes = [PyInPortVectorDense]

        for cls in classes:
            self.probe_test_routine(cls)


class MockCspPort(AbstractCspPort):
    @property
    def name(self) -> str:
        return "mock_csp_port"

    @property
    def d_type(self) -> np.dtype:
        return np.int32.dtype

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return (4, 3, 2)

    @property
    def size(self) -> int:
        return 24

    def start(self):
        pass

    def join(self):
        pass


class TestVirtualPortTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.csp_ports = {"id0": MockCspPort(),
                          "id1": MockCspPort()}
        self.transform_funcs = {"id0": [ft.partial(lambda x, y: x + y, 1),
                                        ft.partial(lambda x, y: x + y, 1)],
                                "id1": [ft.partial(lambda x, y: x * y, 2)]}

    def test_init(self) -> None:
        """Tests the initialization of a VirtualPortTransformer."""
        vpt = VirtualPortTransformer(self.csp_ports,
                                     self.transform_funcs)

        self.assertIsInstance(vpt, VirtualPortTransformer)
        self.assertEqual(vpt._csp_port_to_fp,
                         {self.csp_ports[i]: self.transform_funcs[i]
                          for i in ["id0", "id1"]})

    def test_empty_csp_ports_raises_exception(self) -> None:
        """Tests whether an exception is raised when the specified CSP ports
        are empty."""
        with self.assertRaises(AssertionError):
            VirtualPortTransformer({}, self.transform_funcs)

    def test_transform(self) -> None:
        """Tests whether the transformation produces the correct
        results depending on which CSP port is specified."""
        vpt = VirtualPortTransformer(self.csp_ports,
                                     self.transform_funcs)
        data = np.array(5)
        self.assertEqual(vpt.transform(data, self.csp_ports["id0"]), data + 2)
        self.assertEqual(vpt.transform(data, self.csp_ports["id1"]), data * 2)

    def test_transformation_defaults_to_identity(self) -> None:
        """Tests whether the transformation defaults to the identity
        transformation when no transformation functions are specified."""
        del(self.transform_funcs["id0"])
        vpt = VirtualPortTransformer(self.csp_ports,
                                     self.transform_funcs)
        data = np.array(5)
        self.assertEqual(vpt.transform(data, self.csp_ports["id0"]), data)
        self.assertEqual(vpt.transform(data, self.csp_ports["id1"]), data * 2)

    def test_transformation_with_unknown_csp_port_raises_exception(self) -> \
            None:
        """Tests whether an exception is raised when transform() is called
        with an unknown CSP port."""
        vpt = VirtualPortTransformer(self.csp_ports,
                                     self.transform_funcs)
        with self.assertRaises(AssertionError):
            vpt.transform(np.array(5), MockCspPort())


class TestIdentityTransformer(unittest.TestCase):
    def test_init(self) -> None:
        """Tests the initialization of an IdentityTransformer."""
        it = IdentityTransformer()
        self.assertIsInstance(it, IdentityTransformer)

    def test_transform(self) -> None:
        """Tests whether the transformation is the identity transformation."""
        it = IdentityTransformer()
        data = np.array(5)
        self.assertEqual(it.transform(data, MockCspPort()), data)


if __name__ == '__main__':
    unittest.main()
