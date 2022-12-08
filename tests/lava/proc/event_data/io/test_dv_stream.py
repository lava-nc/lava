# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.io.dv_stream import DvStream, DvStreamPM


class TestProcessDvStream(unittest.TestCase):
    def test_init(self) -> None:
        """Tests instantiation of DvStream."""
        stream = DvStream(address="127.0.0.1",
                          port=7777,
                          shape_frame_in=(35, 35),
                          shape_out=(43200,),
                          additional_kwarg=5)

        self.assertIsInstance(stream, DvStream)
        self.assertEqual(stream.out_port.shape, (43200,))
        self.assertEqual(stream.proc_params["additional_kwarg"], 5)

    def test_invalid_out_shape_throws_exception(self) -> None:
        """Tests whether a shape that is invalid (not one-dimensional) throws
        an exception."""
        invalid_shape = (240, 180)
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_frame_in=(35, 35),
                     shape_out=invalid_shape)

    def test_invalid_in_shape_throws_exception(self) -> None:
        """Tests whether a shape that is invalid (not two-dimensional) throws
        an exception."""
        invalid_in_shape = (240,)
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_frame_in=invalid_in_shape,
                     shape_out=(43200,))

    def test_negative_frame_size_throws_exception(self) -> None:
        """Tests whether a shape with a negative size throws an exception."""
        invalid_shape = (-35,-35)
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_frame_in=invalid_shape,
                     shape_out=(43200,))

    def test_negative_size_throws_exception(self) -> None:
        """Tests whether a shape with a negative size throws an exception."""
        invalid_shape = (-43200,)
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=7777,
                     shape_frame_in=(35, 35),
                     shape_out=invalid_shape)

    def test_negative_port_throws_exception(self) -> None:
        """Tests whether a negative port throws an exception."""
        min_port = 0
        invalid_port = min_port - 1
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=invalid_port,
                     shape_frame_in=(35, 35),
                     shape_out=(43200,))

    def test_port_out_of_range_throws_exception(self) -> None:
        """Tests whether a positive port that is too large throws an
        exception."""
        max_port = 65535
        invalid_port = max_port + 1
        with(self.assertRaises(ValueError)):
            DvStream(address="127.0.0.1",
                     port=invalid_port,
                     shape_frame_in=(35, 35),
                     shape_out=(43200,))

    def test_address_empty_string_throws_exception(self) -> None:
        """Tests whether an empty address throws an exception."""
        invalid_address = ""
        with(self.assertRaises(ValueError)):
            DvStream(address=invalid_address,
                     port=7777,
                     shape_frame_in=(35, 35),
                     shape_out=(43200,))


class RecvSparse(AbstractProcess):
    """Process that receives arbitrary sparse data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Vars.
    """

    def __init__(self,
                 shape: ty.Tuple[int]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))
        self.idx = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvSparse, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvSparsePM(PyLoihiProcessModel):
    """Receives sparse data from PyInPort and stores a padded version of
    received data and indices in Vars."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)
    idx: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data, idx = self.in_port.recv()

        self.data = np.pad(data,
                           pad_width=(0, self.in_port.shape[0] - data.shape[0]))
        self.idx = np.pad(idx,
                          pad_width=(0, self.in_port.shape[0] - data.shape[0]))


class TestProcessModelDvStream(unittest.TestCase):
    def setUp(self) -> None:
        self._mock_packet_input = ({"x": np.asarray([35, 32, 33]), "y": np.asarray([35, 32, 31]),
                                    "polarity": np.asarray([0, 0, 0])},
                                   {"x": np.asarray([35, 32, 33]), "y": np.asarray([35, 32, 31]),
                                    "polarity": np.asarray([0, 0, 0])},
                                   {"x": np.asarray([35, 32, 33]), "y": np.asarray([35, 32, 31]),
                                    "polarity": np.asarray([0, 0, 0])})

        self.proc_params = {
            "address": "127.0.0.1",
            "port": 7777,
            "shape_frame_in": (40, 40),
            "shape_out": (43200,),
            "event_stream": iter(self._mock_packet_input)
        }

    def test_init(self) -> None:
        """Tests instantiation of the DvStream PyProcModel."""
        pm = DvStreamPM(proc_params=self.proc_params)
        self.assertIsInstance(pm, DvStreamPM)

    def test_run_spk_without_subsampling(self) -> None:
        """ Test that run_spk works as expected when no subsampling is needed."""
        max_num_events = 15
        shape = (max_num_events,)
        shape_frame_in = (40,40)
        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=shape,
                             shape_frame_in=shape_frame_in,
                             event_stream=iter(self._mock_packet_input))
        recv_sparse = RecvSparse(shape=shape)

        dv_stream.out_port.connect(recv_sparse.in_port)

        for mock_package in self._mock_packet_input:
            dv_stream.run(condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg())

            expected_data = mock_package["polarity"]
            expected_indices = np.ravel_multi_index((mock_package["x"], mock_package["y"]), shape_frame_in)

            sent_and_received_data = \
                recv_sparse.data.get()[:expected_data.shape[0]]
            sent_and_received_indices = \
                recv_sparse.idx.get()[:expected_indices.shape[0]]

            np.testing.assert_equal(sent_and_received_data,
                                    expected_data)
            np.testing.assert_equal(sent_and_received_indices,
                                    expected_indices)
        dv_stream.stop()

    def test_run_spk_with_no_next(self) -> None:
        """ Test that warning is raised when no events are arriving."""
        # TODO: Check whether warning arrives
        # with (self.assertWarns(UserWarning)):
        max_num_events = 15
        shape = (max_num_events,)

        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=shape,
                             shape_frame_in=(40, 40),
                             event_stream=iter(self._mock_packet_input))
        recv_sparse = RecvSparse(shape=shape)

        dv_stream.out_port.connect(recv_sparse.in_port)
        dv_stream.run(condition=RunSteps(num_steps=len(self._mock_packet_input)+1), run_cfg=Loihi1SimCfg())
        dv_stream.stop()

if __name__ == '__main__':
    unittest.main()

