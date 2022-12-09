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


class MockPacketInput:
    def __init__(self,
                 mock_packets):
        self._mock_packets = mock_packets
        self.timestep = 0

    def __next__(self):
        # if self.timestep < len(self._mock_packets):
        packet = self._mock_packets[self.timestep]
        self.timestep += 1

        return packet
        # else:
        #     raise StopIteration # TODO: change this to actual behavior from DV network input object

    @property
    def mock_packets(self):
        return self._mock_packets

class TestProcessModelDvStream(unittest.TestCase):
    def test_init(self) -> None:
        """Tests instantiation of the DvStream PyProcModel."""
        mock_packets = ({"x": np.asarray([8, 12, 13]), "y": np.asarray([157, 148, 146]),
                         "polarity": np.asarray([0, 1, 0])},
                        {"x": np.asarray([39]), "y": np.asarray([118]),
                         "polarity": np.asarray([1])},
                        {"x": np.asarray([12, 10]), "y": np.asarray([163, 108]),
                         "polarity": np.asarray([1, 1])})

        mock_packet_input = MockPacketInput(mock_packets)

        proc_params = {
            "address": "127.0.0.1",
            "port": 7777,
            "shape_frame_in": (240, 180),
            "shape_out": (43200,),
            "seed_sub_sampling": 0,
            "event_stream": iter(mock_packet_input.mock_packets)
        }

        pm = DvStreamPM(proc_params=proc_params)
        self.assertIsInstance(pm, DvStreamPM)

    def test_run_spk_without_subsampling(self) -> None:
        """
        Tests that run_spk works as expected when no subsampling is needed.
        """
        mock_packets = ({"x": np.asarray([8, 12, 13]), "y": np.asarray([157, 148, 146]),
                         "polarity": np.asarray([0, 1, 0])},
                        {"x": np.asarray([39]), "y": np.asarray([118]),
                         "polarity": np.asarray([1])},
                        {"x": np.asarray([12, 10]), "y": np.asarray([163, 108]),
                         "polarity": np.asarray([1, 1])})

        mock_packet_input = MockPacketInput(mock_packets)

        # data and indices calculated from the mock packets
        data_history = [
            [0, 1, 0],
            [1],
            [1, 1]
        ]
        indices_history = [
            [1597, 2308, 2486],
            [7138],
            [2323, 1908]
        ]

        max_num_events = 15
        shape_frame_in = (240, 180)
        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=(max_num_events,),
                             shape_frame_in=shape_frame_in,
                             event_stream=iter(mock_packet_input.mock_packets))

        recv_sparse = RecvSparse(shape=(max_num_events,))

        dv_stream.out_port.connect(recv_sparse.in_port)

        num_steps = 3
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            dv_stream.run(condition=run_cnd, run_cfg=run_cfg)

            expected_data = np.array(data_history[i])
            expected_indices = np.array(indices_history[i])

            received_data = \
                recv_sparse.data.get()[:expected_data.shape[0]]
            received_indices = \
                recv_sparse.idx.get()[:expected_indices.shape[0]]

            np.testing.assert_equal(received_data, expected_data)
            np.testing.assert_equal(received_indices, expected_indices)

        dv_stream.stop()

    def test_run_spk_with_empty_batch(self) -> None:
        """ Test that warning is raised when no events are arriving."""
        # TODO: Add appropriate behavior in process
        mock_packets = ({"x": np.asarray([8, 12, 13]), "y": np.asarray([157, 148, 146]),
                         "polarity": np.asarray([0, 1, 0])},
                        {"x": np.asarray([39]), "y": np.asarray([118]),
                         "polarity": np.asarray([1])},
                        {"x": np.asarray([12, 10]), "y": np.asarray([163, 108]),
                         "polarity": np.asarray([1, 1])},
                        {"x": np.asarray([]), "y": np.asarray([]),
                         "polarity": np.asarray([])})

        mock_packet_input = MockPacketInput(mock_packets)

        max_num_events = 15
        shape_frame_in = (240, 180)
        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=(max_num_events,),
                             shape_frame_in=shape_frame_in,
                             event_stream=iter(mock_packet_input.mock_packets))

        recv_sparse = RecvSparse(shape=(max_num_events,))

        dv_stream.out_port.connect(recv_sparse.in_port)

        num_steps = 4
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            print(i)
            dv_stream.run(condition=run_cnd, run_cfg=run_cfg)

        dv_stream.stop()

    def test_run_spk_with_no_batch(self) -> None:
        """ Test that an exception is thrown when the event stream stops."""
        # TODO: Add behavior in dv_stream
        # with (self.assertWarns(UserWarning)):
        mock_packets = ({"x": np.asarray([8, 12, 13]), "y": np.asarray([157, 148, 146]),
                         "polarity": np.asarray([0, 1, 0])},)

        mock_packet_input = MockPacketInput(mock_packets)

        max_num_events = 15
        shape_frame_in = (240, 180)
        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=(max_num_events,),
                             shape_frame_in=shape_frame_in,
                             event_stream=iter(mock_packet_input.mock_packets))

        recv_sparse = RecvSparse(shape=(max_num_events,))

        dv_stream.out_port.connect(recv_sparse.in_port)

        num_steps = len(mock_packets) + 1
        print(num_steps)
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            print(i)
            dv_stream.run(condition=run_cnd, run_cfg=run_cfg)

        dv_stream.stop()

    def test_run_spk_with_sub_sampling(self):
        mock_packets = ({"x": np.asarray([8, 12, 13, 13, 13, 9, 14, 14, 13, 13, 8, 9, 9, 13, 9]),
                         "y": np.asarray([157, 148, 146, 156, 158, 167, 122, 113, 149, 148, 156,
                                          109, 107, 160, 160]),
                         "polarity": np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])},
                        {"x": np.asarray([39]),
                         "y": np.asarray([118]),
                         "polarity": np.asarray([1])})

        self.mock_packet_input = MockPacketInput(mock_packets)

        # data and indices calculated from the mock packets
        expected_data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1]
        ]
        expected_indices = [
            [1597., 2486., 2496., 2498., 1787., 2633., 1729., 1727., 2500., 1780.],
            [7138]
        ]

        max_num_events = 10
        shape_frame_in = (240, 180)
        seed_rng = 0
        dv_stream = DvStream(address="127.0.0.1",
                             port=7777,
                             shape_out=(max_num_events,),
                             shape_frame_in=shape_frame_in,
                             event_stream=iter(self.mock_packet_input.mock_packets),
                             seed_sub_sampling=seed_rng)

        recv_sparse = RecvSparse(shape=(max_num_events,))

        dv_stream.out_port.connect(recv_sparse.in_port)

        num_steps = 2
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            dv_stream.run(condition=run_cnd, run_cfg=run_cfg)

            received_data = \
                recv_sparse.data.get()[:len(expected_data[i])]
            received_indices = \
                recv_sparse.idx.get()[:len(expected_indices[i])]

            np.testing.assert_equal(received_data, expected_data[i])
            np.testing.assert_equal(received_indices, expected_indices[i])

        dv_stream.stop()


if __name__ == '__main__':
    unittest.main()

