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
from lava.proc.event_data.events_to_frame.process import EventsToFrame
from lava.proc.event_data.events_to_frame.models import PyEventsToFramePM


class RecvDense(AbstractProcess):
    """Process that receives arbitrary dense data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    """
    def __init__(self,
                 shape: ty.Union[
                     ty.Tuple[int, int], ty.Tuple[int, int, int]]) -> None:
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


class SendSparse(AbstractProcess):
    """Process that sends arbitrary sparse data.

    Parameters
    ----------
    shape: tuple
        Shape of the OutPort.
    """
    def __init__(self,
                 shape: ty.Tuple[int],
                 data: np.ndarray,
                 indices: np.ndarray) -> None:
        super().__init__(shape=shape, data=data, indices=indices)

        self.out_port = OutPort(shape=shape)


@implements(proc=SendSparse, protocol=LoihiProtocol)
@requires(CPU)
class PySendSparsePM(PyLoihiProcessModel):
    """Sends sparse data to PyOutPort."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._data = proc_params["data"]
        self._indices = proc_params["indices"]

    def run_spk(self) -> None:
        data = self._data
        idx = self._indices

        self.out_port.send(data, idx)


class TestProcessModelEventsEventsToFrame(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the SparseToDense process model."""
        pm = PyEventsToFramePM()

        self.assertIsInstance(pm, PyEventsToFramePM)

    def test_third_dimension_1(self):
        data = np.array([1, 1, 1, 1, 1, 1])
        xs = [0, 1, 2, 1, 2, 4]
        ys = [0, 2, 1, 5, 7, 7]
        indices = np.ravel_multi_index((xs, ys), (8, 8))

        expected_data = np.zeros((8, 8, 1))
        expected_data[0, 0, 0] = 1
        expected_data[1, 2, 0] = 1
        expected_data[2, 1, 0] = 1

        expected_data[1, 5, 0] = 1
        expected_data[2, 7, 0] = 1

        expected_data[4, 7, 0] = 1

        send_sparse = SendSparse(shape=(10,), data=data, indices=indices)
        to_frame = EventsToFrame(shape_in=(10,),
                                 shape_out=(8, 8, 1))
        recv_dense = RecvDense(shape=(8, 8, 1))

        send_sparse.out_port.connect(to_frame.in_port)
        to_frame.out_port.connect(recv_dense.in_port)

        num_steps = 1
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        to_frame.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        to_frame.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)

    def test_third_dimension_2(self):
        data = np.array([1, 0, 1, 0, 1, 0])
        xs = [0, 1, 2, 1, 2, 4]
        ys = [0, 2, 1, 5, 7, 7]
        indices = np.ravel_multi_index((xs, ys), (8, 8))

        expected_data = np.zeros((8, 8, 2))
        expected_data[0, 0, 1] = 1
        expected_data[1, 2, 0] = 1
        expected_data[2, 1, 1] = 1

        expected_data[1, 5, 0] = 1
        expected_data[2, 7, 1] = 1

        expected_data[4, 7, 0] = 1

        send_sparse = SendSparse(shape=(10,), data=data, indices=indices)
        to_frame = EventsToFrame(shape_in=(10,),
                                 shape_out=(8, 8, 2))
        recv_dense = RecvDense(shape=(8, 8, 2))

        send_sparse.out_port.connect(to_frame.in_port)
        to_frame.out_port.connect(recv_dense.in_port)

        num_steps = 1
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        to_frame.run(condition=run_cnd, run_cfg=run_cfg)

        sent_and_received_data = \
            recv_dense.data.get()

        to_frame.stop()

        np.testing.assert_equal(sent_and_received_data,
                                expected_data)


if __name__ == '__main__':
    unittest.main()
