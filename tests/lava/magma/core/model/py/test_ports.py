# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.sparse as sparse

from lava.magma.compiler.channels.interfaces import (
    AbstractCspSendPort,
    AbstractCspRecvPort
)
from lava.magma.core.model.interfaces import PortMessageFormat
from lava.magma.core.model.py.ports import (
    PyInPort,
    PyOutPort,
    PyInPortVectorDense,
    PyInPortVectorSparse,
    PyInPortScalarDense,
    PyInPortScalarSparse,
    PyOutPortVectorDense,
    PyOutPortVectorSparse,
    PyOutPortScalarDense,
    PyOutPortScalarSparse,
    PyPortMessage
)


from lava.magma.compiler.channels.pypychannel import PyPyChannel


# TODO: Need to test the process_builder.build flow which creates the
# TODO: inport and outports within the function.


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

    """ def create_message(fmt, n_elem, data) -> PyPortMessage:
    msg_header = PyPortMessageHeader(fmt, n_elem)
    msg_payload = PyPortMessagePayload(data)
    return """


class TestPyPorts(unittest.TestCase):
    """
    """

    def test_pyport_send_vec_dense_to_vec_dense(self):
        """Tests sending data over a dense vector outport to a dense vector
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    def test_pyport_send_vec_dense_to_vec_sparse(self):
        """Tests sending data over a dense vector outport to a sparse vector
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix
            sparse_data = sparse.csr_matrix(data)

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data, recv_idx = recv_py_port.recv()

            print("")
            print("If working, below arrays are np.array_equal")
            print(sparse_data.data)
            print("")
            print(sparse_data.indices)
            print(" equals: ")
            print(recv_data)
            print("")
            print(recv_idx)
            assert np.array_equal(sparse_data.data, recv_data)
            assert np.array_equal(sparse_data.indices, recv_idx)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_vec_dense_to_sca_dense(self):
        """Tests sending data over a dense vector outport to a dense scalar
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_vec_dense_to_sca_sparse(self):
        """Tests sending data over a dense vector outport to a sparse scalar
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.VECTOR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    def test_pyport_send_vec_sparse_to_vec_sparse(self):
        """Tests sending data over a sparse vector outport to a sparse vector
        inport"""
        idx = [None] * 2
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = sparse.random(5, 5, density=0.05)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            idx[0], idx[1], sparse_data = sparse.find(
                sparse_matrix
            )

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(sparse_data, idx)
            recv_idx, recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(sparse_data)
            print(" ")
            print([idx[0][0], idx[1][0]])
            print(" equals: ")
            print(recv_data)
            print(" ")
            print(list(map(int, recv_idx)))
            assert np.array_equal(sparse_data, recv_data)
            assert np.array_equal([idx[0][0], idx[1][0]],
                                  list(map(int, recv_idx)))
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_vec_sparse_to_vec_dense(self):
        """Tests sending data over a sparse vector outport to a dense vector
        inport"""
        idx = [[], []]
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = sparse.random(5, 5, density=0.05)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data, idx[0], idx[1] = sparse.find(
                sparse_matrix
            )

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_vec_sparse_to_sca_dense(self):
        """Tests sending data over a sparse vector outport to a dense scalar
        inport"""
        idx = [[], []]
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = sparse.random(5, 5, density=0.05)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data, idx[0], idx[1] = sparse.find(
                sparse_matrix
            )

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_vec_sparse_to_sca_sparse(self):
        """Tests sending data over a sparse vector outport to a sparse scalar
        inport"""
        idx = [[], []]
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = sparse.random(5, 5, density=0.05)
            message = PyPortMessage(
                PortMessageFormat.VECTOR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data, idx[0], idx[1] = sparse.find(
                sparse_matrix
            )

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortVectorSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_dense_to_sca_dense(self):
        """Tests sending data over a dense scalar outport to a dense scalar
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(1, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_dense_to_vec_dense(self):
        """Tests sending data over a dense scalar outport to a dense vector
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(1, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_dense_to_sca_sparse(self):
        """Tests sending data over a dense scalar outport to a sparse scalar
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(1, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    def test_pyport_send_sca_dense_to_vec_sparse(self):
        """Tests sending data over a dense scalar outport to a sparse vector
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            dense_matrix = np.random.randint(5, size=(1, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_DENSE,
                dense_matrix.size,
                dense_matrix
            )
            data = dense_matrix
            sparse_data = sparse.csr_matrix(data)

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarDense(process_model=None,
                                     csp_ports=send_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data, recv_idx = recv_py_port.recv()

            print("")
            print("If working, below arrays are np.array_equal")
            print(sparse_data.data)
            print("")
            print(sparse_data.indices)
            print(" equals: ")
            print(recv_data)
            print("")
            print(recv_idx)
            assert np.array_equal(sparse_data.data, recv_data)
            assert np.array_equal(sparse_data.indices, recv_idx)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_sparse_to_sca_sparse(self):
        """Tests sending data over a scalar sparse outport to a scalar sparse
        inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data = sparse_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_sparse_to_sca_dense(self):
        """Tests sending data over a sparse scalar outport to a
        dense scalar inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data = sparse_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortScalarDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_sparse_to_vec_dense(self):
        """Tests sending data over a sparse scalar outport
        to dense vector inport"""
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            data = sparse_matrix

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorDense(process_model=None,
                                    csp_ports=recv_csp_port,
                                    shape=message._payload.shape,
                                    d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(data)
            recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(data)
            print(" equals: ")
            print(recv_data)
            assert np.array_equal(data, recv_data)
        finally:
            smm.shutdown()

    @unittest.SkipTest
    def test_pyport_send_sca_sparse_to_vec_sparse(self):
        """Tests sending data over a sparse scalar outport
        to sparse vector inport"""
        idx = [None] * 2
        smm = SharedMemoryManager()
        try:
            smm.start()

            sparse_matrix = np.random.randint(5, size=(2, 4))
            message = PyPortMessage(
                PortMessageFormat.SCALAR_SPARSE,
                sparse_matrix.size,
                sparse_matrix
            )
            idx[0], idx[1], sparse_data = sparse.find(
                sparse_matrix
            )

            channel = get_channel(smm,
                                  message._payload,
                                  size=message._payload.size)

            send_csp_port: AbstractCspSendPort = channel.src_port
            recv_csp_port: AbstractCspRecvPort = channel.dst_port

            send_py_port: PyOutPort = \
                PyOutPortScalarSparse(process_model=None,
                                      csp_ports=send_csp_port,
                                      shape=message._payload.shape,
                                      d_type=message._payload.dtype)
            recv_py_port: PyInPort = \
                PyInPortVectorSparse(process_model=None,
                                     csp_ports=recv_csp_port,
                                     shape=message._payload.shape,
                                     d_type=message._payload.dtype)

            recv_py_port.start()
            send_py_port.start()

            send_py_port.send(sparse_data, idx)
            recv_idx, recv_data = recv_py_port.recv()

            print("If working, below arrays are np.array_equal")
            print(sparse_data)
            print(" ")
            print([idx[0][0], idx[1][0]])
            print(" equals: ")
            print(recv_data)
            print(" ")
            print(list(map(int, recv_idx)))
            assert np.array_equal(sparse_data, recv_data)
            assert np.array_equal([idx[0][0], idx[1][0]],
                                  list(map(int, recv_idx)))
        finally:
            smm.shutdown()


if __name__ == '__main__':
    unittest.main()
