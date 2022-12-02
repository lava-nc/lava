# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
from lava.proc.event_data.event_pre_processor.dense_to_dense.down_sampling_dense import DownSamplingDense, DownSamplingDensePM
from lava.proc.event_data.event_pre_processor.utils import DownSamplingMethodDense

import numpy as np

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

import matplotlib.pyplot as plt


class RecvDense(AbstractProcess):
    def __init__(self,
                 shape: tuple) -> None:
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
                 shape: tuple,
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


class TestProcessDownSamplingDense(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of DownSamplingDense."""
        down_sampler = DownSamplingDense(shape_in=(240, 180),
                                         down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                         down_sampling_factor=8)

        self.assertIsInstance(down_sampler, DownSamplingDense)
        self.assertEqual(down_sampler.proc_params["shape_in"], (240, 180))
        self.assertEqual(down_sampler.proc_params["down_sampling_method"], DownSamplingMethodDense.CONVOLUTION)
        self.assertEqual(down_sampler.proc_params["down_sampling_factor"], 8)

    def test_invalid_shape_in_negative_width_or_height(self):
        """Checks if an error is raised when a negative width or height
        for shape_in is given."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(-240, 180),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, -180),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

    def test_invalid_shape_in_decimal_width_or_height(self):
        """Checks if an error is raised when a decimal width or height
        for shape_in is given."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240.5, 180),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, 180.5),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

    def test_invalid_shape_in_dimension(self):
        """Checks if an error is raised when a 1d or 4d input shape is given."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240,),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, 180, 2, 1),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

    def test_invalid_shape_in_third_dimension_not_2(self):
        """Checks if an error is raised if the value of the 3rd dimension
        for the shape_in parameter is not 2."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, 180, 1),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8)

    def test_invalid_down_sampling_factor_negative(self):
        """Checks if an error is raised if the given down sampling factor
        is negative."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, 180),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=-8)

    def test_invalid_down_sampling_factor_decimal(self):
        """Checks if an error is raised if the given down sampling factor is decimal."""
        with(self.assertRaises(ValueError)):
            _ = DownSamplingDense(shape_in=(240, 180),
                                  down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                  down_sampling_factor=8.5)

    def test_invalid_down_sampling_method(self):
        """Checks if an error is raised if the given down sampling method is not of type
        DownSamplingMethodDense."""
        with(self.assertRaises(TypeError)):
            _ = DownSamplingDense(shape_in=(240, 180),
                                  down_sampling_method="convolution",
                                  down_sampling_factor=8)


# TODO (GK): Add tests for widths and heights not divisible by
# TODO (GK): down_sampling_factor
class TestProcessModelDownSamplingDense(unittest.TestCase):
    def test_init(self):
        proc_params = {
            "shape_in": (240, 180),
            "down_sampling_method": DownSamplingMethodDense.SKIPPING,
            "down_sampling_factor": 8
        }

        pm = DownSamplingDensePM(proc_params)

        self.assertIsInstance(pm, DownSamplingDensePM)
        self.assertEqual(pm._shape_in, proc_params["shape_in"])
        self.assertEqual(pm._down_sampling_method,
                         proc_params["down_sampling_method"])
        self.assertEqual(pm._down_sampling_factor,
                         proc_params["down_sampling_factor"])

    def test_run(self):
        data = np.zeros((8, 8))

        send_dense = SendDense(shape=(8, 8), data=data)
        down_sampler = DownSamplingDense(shape_in=(8, 8),
                                         down_sampling_method=DownSamplingMethodDense.SKIPPING,
                                         down_sampling_factor=2)

        send_dense.out_port.connect(down_sampler.in_port)

        # Run parameters
        num_steps = 1
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        # Running
        down_sampler.run(condition=run_cnd, run_cfg=run_cfg)

        # Stopping
        down_sampler.stop()

        self.assertFalse(down_sampler.runtime._is_running)

    def test_down_sampling_skipping(self):
        data = np.zeros((8, 8))
        data[0, 0] = 1
        data[1, 2] = 1
        data[2, 1] = 1

        data[1, 5] = 1
        data[2, 7] = 1

        data[4, 4] = 1

        expected_data = np.zeros((2, 2))
        expected_data[0, 0] = 1
        expected_data[1, 1] = 1

        send_dense = SendDense(shape=(8, 8), data=data)
        down_sampler = DownSamplingDense(shape_in=(8, 8),
                                         down_sampling_method=DownSamplingMethodDense.SKIPPING,
                                         down_sampling_factor=4)
        recv_dense = RecvDense(shape=(2, 2))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

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

    def test_down_sampling_max_pooling(self):
        data = np.zeros((8, 8))
        data[0, 0] = 1
        data[1, 2] = 1
        data[2, 1] = 1

        data[1, 5] = 1
        data[2, 7] = 1

        data[4, 4] = 1

        expected_data = np.zeros((2, 2))
        expected_data[0, 0] = 1
        expected_data[0, 1] = 1
        expected_data[1, 1] = 1

        send_dense = SendDense(shape=(8, 8), data=data)
        down_sampler = DownSamplingDense(shape_in=(8, 8),
                                         down_sampling_method=DownSamplingMethodDense.MAX_POOLING,
                                         down_sampling_factor=4)
        recv_dense = RecvDense(shape=(2, 2))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

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

    def test_down_sampling_convolution(self):
        data = np.zeros((8, 8))
        data[0, 0] = 1
        data[1, 2] = 1
        data[2, 1] = 1

        data[1, 5] = 1
        data[2, 7] = 1

        data[4, 4] = 1

        expected_data = np.zeros((2, 2))
        expected_data[0, 0] = 3
        expected_data[0, 1] = 2
        expected_data[1, 1] = 1

        send_dense = SendDense(shape=(8, 8), data=data)
        down_sampler = DownSamplingDense(shape_in=(8, 8),
                                         down_sampling_method=DownSamplingMethodDense.CONVOLUTION,
                                         down_sampling_factor=4)
        recv_dense = RecvDense(shape=(2, 2))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        # Run parameters
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        # TODO : REMOVE THIS AFTER DEBUG
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Max pooling')
        ax1.imshow(data)
        ax1.set_title("Data")
        ax2.imshow(expected_data)
        ax2.set_title("Expected data")
        ax3.imshow(sent_and_received_data)
        ax3.set_title("Actual data")
        fig.show()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)


if __name__ == '__main__':
    unittest.main()
