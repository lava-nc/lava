# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.sparse as sparse

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.core.model.interfaces import PortMessageFormat
from lava.magma.core.model.py.ports import PyInPort, PyInPortVectorDense, \
    PyOutPortVectorDense, PyOutPort, PyPortMessage, \
    PyPortMessageHeader, PyPortMessagePayload


from lava.magma.compiler.channels.pypychannel import PyPyChannel


def get_channel(smm, data, size, name="test_channel") -> PyPyChannel:
    return PyPyChannel(
        smm=smm, src_name=name, dst_name=name, shape=data.shape,
        dtype=data.dtype, size=size
    )


def create_message(fmt, n_elem, data) -> PyPortMessage:
    msg_header = PyPortMessageHeader(fmt, n_elem)
    msg_payload = PyPortMessagePayload(data)
    return PyPortMessage(msg_header,
                         msg_payload)


class TestPyPorts(unittest.TestCase):

    def message_creation():
        return create_message(
            PortMessageFormat.VECTOR_DENSE,
            2,
            np.random.randint(5, size=(2, 4))
        ), create_message(
            PortMessageFormat.VECTOR_SPARSE,
            5,
            sparse.random(5, 5, density=0.05)
        ), create_message(
            PortMessageFormat.SCALAR_DENSE,
            4,
            np.random.randint(5, size=(1, 4))
        ), create_message(
            PortMessageFormat.SCALAR_SPARSE,
            4,
            np.random.randint(5, size=(2, 4))
        )

    def test_port_send_out_vec_dense_to_in_vec_dense(self):
        """Tests sending data over a dense vector outport to a dense vector
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            shape = (2, 4)
            d_type = np.int32
            message = create_message(
                PortMessageFormat.VECTOR_DENSE,
                2,
                np.random.randint(5, size=(2, 4))
            )
            channel = get_channel(smm, message.payload(), size=2)

            send_csp_port: AbstractCspPort = channel.src_port
            recv_csp_port: AbstractCspPort = channel.dst_port

            recv_py_port: PyInPort = \
                PyInPortVectorDense(process_model=None,
                                    csp_port=recv_csp_port,
                                    shape=shape,
                                    d_type=d_type)
            send_py_port: PyOutPort = \
                PyOutPortVectorDense(process_model=None,
                                     csp_port=send_csp_port,
                                     shape=shape,
                                     d_type=d_type)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(message)

            assert np.array_equal(message.payload(), recv_py_port.recv())
        finally:
            smm.shutdown()

    # TODO: Need to test the process_builder.build flow which creates the
    # TODO: inport and outports within the function.


if __name__ == '__main__':
    unittest.main()
