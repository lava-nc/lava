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
from lava.proc.event_data.event_pre_processor.dense_to_dense.flattening import Flattening, FlatteningPM

# TODO: add doc strings for these processes
class RecvDense(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvDense, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvDensePM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data = self.in_port.recv()

        self.data = data


class SendDense(AbstractProcess):
    def __init__(self,
                 shape: ty.Union[ty.Tuple[int, int], ty.Tuple[int, int, int]],
                 data: np.ndarray) -> None:
        super().__init__(shape=shape, data=data)

        self.out_port = OutPort(shape=shape)


@implements(proc=SendDense, protocol=LoihiProtocol)
@requires(CPU)
class PySendDensePM(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._data = proc_params["data"]

    def run_spk(self) -> None:
        data = self._data

        self.out_port.send(data)

class TestProcessFlattening(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of DownSamplingDense."""
        flattener = Flattening(shape_in=(240, 180))

        self.assertIsInstance(flattener, Flattening)
        self.assertEqual(flattener.proc_params["shape_in"], (240, 180))

    def test_negative_width_or_height_throws_exception(self):
        """Tests whether an exception is thrown when a negative width or height for the shape_in argument is given."""
        with(self.assertRaises(ValueError)):
            Flattening(shape_in=(-240, 180))

        with(self.assertRaises(ValueError)):
            Flattening(shape_in=(240, -180))

    def test_too_few_or_too_many_dimensions_throws_exception(self):
        """Tests whether an exception is thrown when a 1d or 4d value for the shape_in argument is given."""
        with(self.assertRaises(ValueError)):
            Flattening(shape_in=(240,))

        with(self.assertRaises(ValueError)):
            Flattening(shape_in=(240, 180, 2, 1))

    def test_third_dimension_not_2_throws_exception(self):
        """Tests whether an exception is thrown if the value of the 3rd dimension for the shape_in argument is not 2."""
        with(self.assertRaises(ValueError)):
            Flattening(shape_in=(240, 180, 1))

# TODO: add doc strings
class TestProcessModelFlattening(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the Flattening process model"""
        proc_params = {
            "shape_in": (240, 180)
        }

        pm = FlatteningPM(proc_params)

        self.assertIsInstance(pm, FlatteningPM)
        self.assertEqual(pm._shape_in, proc_params["shape_in"])

    # TODO: can probably be deleted
    def test_run(self):
        data = np.zeros((8, 8))

        send_dense = SendDense(shape=(8, 8), data=data)
        flattener = Flattening(shape_in=(8, 8))

        send_dense.out_port.connect(flattener.in_port)

        # Run parameters
        num_steps = 1
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        # Running
        flattener.run(condition=run_cnd, run_cfg=run_cfg)

        # Stopping
        flattener.stop()

        self.assertFalse(flattener.runtime._is_running)

    def test_flattening_2d(self):
        data = np.zeros((8, 8))

        expected_data = np.zeros((64,))

        send_dense = SendDense(shape=(8, 8), data=data)
        flattener = Flattening(shape_in=(8, 8))
        recv_dense = RecvDense(shape=(64,))

        send_dense.out_port.connect(flattener.in_port)
        flattener.out_port.connect(recv_dense.in_port)

        # Run parameters
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_flattening_3d(self):
        data = np.zeros((8, 8, 2))

        expected_data = np.zeros((128,))

        send_dense = SendDense(shape=(8, 8, 2), data=data)
        flattener = Flattening(shape_in=(8, 8, 2))
        recv_dense = RecvDense(shape=(128,))

        send_dense.out_port.connect(flattener.in_port)
        flattener.out_port.connect(recv_dense.in_port)

        # Run parameters
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)


if __name__ == '__main__':
    unittest.main()
