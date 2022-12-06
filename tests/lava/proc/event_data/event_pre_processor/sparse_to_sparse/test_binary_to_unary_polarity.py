# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.event_pre_processor.sparse_to_sparse.binary_to_unary_polarity import \
    BinaryToUnaryPolarity, BinaryToUnaryPolarityPM

# TODO: add doc strings for these processes
class RecvSparse(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))
        self.idx = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvSparse, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvSparsePM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)
    idx: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data, idx = self.in_port.recv()

        self.data = np.pad(data,
                           pad_width=(
                               0, self.in_port.shape[0] - data.shape[0]))
        self.idx = np.pad(idx,
                          pad_width=(
                              0, self.in_port.shape[0] - data.shape[0]))


class SendSparse(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int],
                 data: np.ndarray,
                 indices: np.ndarray) -> None:
        super().__init__(shape=shape, data=data, indices=indices)

        self.out_port = OutPort(shape=shape)


@implements(proc=SendSparse, protocol=LoihiProtocol)
@requires(CPU)
class PySendSparsePM(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._data = proc_params["data"]
        self._indices = proc_params["indices"]

    def run_spk(self) -> None:
        data = self._data
        idx = self._indices

        self.out_port.send(data, idx)


class TestProcessBinaryToUnaryPolarity(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of BinaryToUnaryPolarity."""
        converter = BinaryToUnaryPolarity(shape=(43200,))

        self.assertIsInstance(converter, BinaryToUnaryPolarity)
        self.assertEqual(converter.proc_params["shape"], (43200,))

    def test_invalid_shape_throws_exception(self):
        """Tests whether a shape argument with an invalid shape throws an exception."""
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=(240, 180))

    def test_negative_size_throws_exception(self):
        """Tests whether a shape argument with a negative size throws an exception."""
        with(self.assertRaises(ValueError)):
            BinaryToUnaryPolarity(shape=(-43200,))

# TODO: add doc strings
class TestProcessModelBinaryToUnaryPolarity(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the BinaryToUnary process model."""
        proc_params = {
            "shape": (10,)
        }

        pm = BinaryToUnaryPolarityPM(proc_params)

        self.assertIsInstance(pm, BinaryToUnaryPolarityPM)

    def test_binary_to_unary_polarity_encoding(self):
        # TODO: add explanations for the meaning of binary and unary somewhere? explain test variables?
        """Tests whether the encoding from binary to unary works correctly."""
        data = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0])
        indices = np.array([1, 5, 4, 3, 3, 2, 0, 1, 0])

        expected_data = data
        expected_data[expected_data == 0] = 1

        expected_indices = indices

        send_sparse = SendSparse(shape=(10, ), data=data, indices=indices)
        binary_to_unary_encoder = BinaryToUnaryPolarity(shape=(10,))
        recv_sparse = RecvSparse(shape=(10, ))

        send_sparse.out_port.connect(binary_to_unary_encoder.in_port)
        binary_to_unary_encoder.out_port.connect(recv_sparse.in_port)

        # Run parameters
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        send_sparse.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_sparse.data.get()[:expected_data.shape[0]]
        sent_and_received_indices = \
            recv_sparse.idx.get()[:expected_indices.shape[0]]

        # Stopping
        send_sparse.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)
        np.testing.assert_equal(sent_and_received_indices,
                                expected_indices)

    # TODO: I guess not needed? should any edge cases be treated?
    def test_run(self):
        data = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0])
        indices = np.array([1, 5, 4, 3, 3, 2, 0, 1, 0])

        send_sparse = SendSparse(shape=(10,), data=data, indices=indices)
        binary_to_unary_encoder = BinaryToUnaryPolarity(shape=(10,))

        send_sparse.out_port.connect(binary_to_unary_encoder.in_port)

        # Run parameters
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        binary_to_unary_encoder.run(condition=run_cnd, run_cfg=run_cfg)

        binary_to_unary_encoder.stop()

        self.assertFalse(binary_to_unary_encoder.runtime._is_running)


if __name__ == '__main__':
    unittest.main()
