# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
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
from lava.proc.max_pooling.process import MaxPooling
from lava.proc.max_pooling.models import PyMaxPoolingPM


class RecvDense(AbstractProcess):
    """Process that receives arbitrary dense data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    """
    def __init__(self,
                 shape: tuple) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvDense, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvDensePM(PyLoihiProcessModel):
    """Receives dense data from PyInPort and stores it in a Var."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data = self.in_port.recv()

        self.data = data


class SendDense(AbstractProcess):
    """Process that sends arbitrary dense data.

    Parameters
    ----------
    shape: tuple
        Shape of the OutPort.
    """
    def __init__(self,
                 shape: tuple,
                 data: np.ndarray) -> None:
        super().__init__(shape=shape, data=data)

        self.out_port = OutPort(shape=shape)


@implements(proc=SendDense, protocol=LoihiProtocol)
@requires(CPU)
class PySendDensePM(PyLoihiProcessModel):
    """Sends dense data to PyOutPort."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._data = proc_params["data"]

    def run_spk(self) -> None:
        data = self._data

        self.out_port.send(data)


class TestProcessModelMaxPooling(unittest.TestCase):
    def test_init(self):
        pm = PyMaxPoolingPM()

        self.assertIsInstance(pm, PyMaxPoolingPM)

    def test_max_pooling(self):
        data = np.zeros((8, 8, 1))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        data[1, 5, 0] = 1
        data[2, 7, 0] = 1

        data[4, 7, 0] = 1

        expected_data = np.zeros((2, 2, 1))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 0] = 1
        expected_data[1, 1, 0] = 1

        send_dense = SendDense(shape=(8, 8, 1), data=data)
        down_sampler = MaxPooling(shape_in=(8, 8, 1),
                                  kernel_size=4)
        recv_dense = RecvDense(shape=(2, 2, 1))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_max_pooling_2_channels(self):
        data = np.zeros((8, 8, 2))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        data[1, 5, 1] = 1
        data[2, 7, 1] = 1

        data[4, 7, 0] = 1

        expected_data = np.zeros((2, 2, 2))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 1] = 1
        expected_data[1, 1, 0] = 1

        send_dense = SendDense(shape=(8, 8, 2), data=data)
        down_sampler = MaxPooling(shape_in=(8, 8, 2),
                                  kernel_size=4)
        recv_dense = RecvDense(shape=(2, 2, 2))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_almost_equal(sent_and_received_data,
                                       expected_data)

    def test_max_pooling_shape_non_divisible_by_kernel_size(self):
        data = np.zeros((9, 9, 1))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        data[1, 5, 0] = 1
        data[2, 7, 0] = 1

        data[4, 8, 0] = 1

        expected_data = np.zeros((2, 2, 1))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 0] = 1

        send_dense = SendDense(shape=(9, 9, 1), data=data)
        down_sampler = MaxPooling(shape_in=(9, 9, 1),
                                  kernel_size=4)
        recv_dense = RecvDense(shape=(2, 2, 1))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_max_pooling_with_padding(self):
        data = np.zeros((4, 4, 1))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        expected_data = np.zeros((2, 2, 1))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 0] = 1
        expected_data[1, 0, 0] = 1

        send_dense = SendDense(shape=(4, 4, 1), data=data)
        down_sampler = MaxPooling(shape_in=(4, 4, 1),
                                  kernel_size=4,
                                  padding=(2, 2))
        recv_dense = RecvDense(shape=(2, 2, 1))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_max_pooling_non_square_kernel(self):
        data = np.zeros((8, 8, 1))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        data[1, 5, 0] = 1
        data[2, 7, 0] = 1

        data[4, 7, 0] = 1

        expected_data = np.zeros((4, 2, 1))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 0] = 1
        expected_data[1, 0, 0] = 1
        expected_data[1, 1, 0] = 1
        expected_data[2, 1, 0] = 1

        send_dense = SendDense(shape=(8, 8, 1), data=data)
        down_sampler = MaxPooling(shape_in=(8, 8, 1),
                                  kernel_size=(2, 4))
        recv_dense = RecvDense(shape=(4, 2, 1))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_max_pooling_stride_different_than_kernel_size(self):
        data = np.zeros((8, 8, 1))
        data[0, 0, 0] = 1
        data[1, 2, 0] = 1
        data[2, 1, 0] = 1

        data[1, 5, 0] = 1
        data[2, 7, 0] = 1

        data[4, 7, 0] = 1

        expected_data = np.zeros((3, 3, 1))
        expected_data[0, 0, 0] = 1
        expected_data[0, 1, 0] = 1
        expected_data[0, 2, 0] = 1
        expected_data[1, 0, 0] = 1
        expected_data[1, 2, 0] = 1
        expected_data[2, 2, 0] = 1

        send_dense = SendDense(shape=(8, 8, 1), data=data)
        down_sampler = MaxPooling(shape_in=(8, 8, 1),
                                  kernel_size=4,
                                  stride=2)
        recv_dense = RecvDense(shape=(3, 3, 1))

        send_dense.out_port.connect(down_sampler.in_port)
        down_sampler.out_port.connect(recv_dense.in_port)

        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        send_dense.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        send_dense.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)


if __name__ == '__main__':
    unittest.main()
