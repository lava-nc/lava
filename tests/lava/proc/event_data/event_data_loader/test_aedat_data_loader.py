# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
from lava.proc.event_data.event_data_loader.aedat_data_loader import AedatDataLoader, \
    AedatDataLoaderPM
from dv import AedatFile
from dv.AedatFile import _AedatFileEventNumpyPacketIterator

import typing as ty
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


class RecvSparse(AbstractProcess):
    def __init__(self,
                 shape: tuple) -> None:
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
                           pad_width=(0, self.in_port.shape[0] - data.shape[0]))
        self.idx = np.pad(idx,
                          pad_width=(0, self.in_port.shape[0] - data.shape[0]))


class TestProcessAedatDataLoader(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of AedatDataLoader"""
        data_loader = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                      shape_out=(43200,))

        self.assertIsInstance(data_loader, AedatDataLoader)
        self.assertEqual(data_loader.proc_params["file_path"],
                         "../dvs_recording.aedat4")
        self.assertEqual(data_loader.proc_params["shape_out"], (43200,))

    def test_invalid_file_path_extension(self):
        """Tests for invalid file extension"""
        with(self.assertRaises(ValueError)):
            # Test for .py
            _ = AedatDataLoader(file_path="test_aedat_data_loader.py",
                                shape_out=(43200,))

    def test_invalid_file_path_missing_file(self):
        """Tests for missing file"""
        with(self.assertRaises(FileNotFoundError)):
            _ = AedatDataLoader(file_path="missing_file.aedat4",
                                shape_out=(43200,))

    def test_invalid_shape_out_dimension(self):
        """Test for an invalid shape given to the shape_out param"""
        with(self.assertRaises(ValueError)):
            _ = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                shape_out=(240, 180))

    def test_invalid_shape_out_negative_integer(self):
        """Tests for a negative width given"""
        with(self.assertRaises(ValueError)):
            _ = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                shape_out=(-43200,))

    def test_invalid_shape_out_decimal(self):
        """Tests for a decimal width given"""
        with(self.assertRaises(ValueError)):
            _ = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                shape_out=(43200.5,))


class TestProcessModelAedatDataLoader(unittest.TestCase):
    def test_init(self):
        proc_params = {
            "file_path": "../dvs_recording.aedat4",
            "shape_out": (3000,),
            "seed_sub_sampling": 0
        }

        pm = AedatDataLoaderPM(proc_params)

        self.assertIsInstance(pm, AedatDataLoaderPM)
        self.assertEqual(pm._shape_out, proc_params["shape_out"])
        self.assertIsInstance(pm._file, AedatFile)
        self.assertIsInstance(pm._stream,
                              _AedatFileEventNumpyPacketIterator)
        self.assertIsInstance(pm._frame_shape, tuple)
        self.assertIsInstance(pm._random_rng, np.random.Generator)

    def test_sub_sampling(self):
        data_history = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0],
            [1, 1, 1],
            [1],
            [1],
            [1]
        ]
        indices_history = [
            [1597, 2308, 2486, 2496, 2498, 1787, 2642, 2633, 2489,
             2488, 1596, 1729, 1727, 2500, 1780],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138, 2301, 2471, 1601, 2982, 1364, 1379, 1386, 1384,
             2983, 1390, 2289, 1401, 1362, 2293],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464],
            [2323, 1908, 1393],
            [4062],
            [1792],
            [3889]
        ]
        seed_rng = 0
        rng = np.random.default_rng(seed=seed_rng)

        max_num_events = 10
        data_loader = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                      shape_out=(max_num_events,),
                                      seed_sub_sampling=seed_rng)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        # Run parameters
        num_steps = 9
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        for i in range(num_steps):
            data_loader.run(condition=run_cnd, run_cfg=run_cfg)

            expected_data = np.array(data_history[i])
            expected_indices = np.array(indices_history[i])

            sent_and_received_data = \
                recv_sparse.data.get()[:expected_data.shape[0]]
            sent_and_received_indices = \
                recv_sparse.idx.get()[:expected_indices.shape[0]]

            if expected_data.shape[0] > max_num_events:
                data_idx_array = np.arange(0, expected_data.shape[0])
                sampled_idx = rng.choice(data_idx_array,
                                         max_num_events,
                                         replace=False)

                expected_data = expected_data[sampled_idx]
                expected_indices = expected_indices[sampled_idx]

            np.testing.assert_equal(sent_and_received_data,
                                    expected_data)
            np.testing.assert_equal(sent_and_received_indices,
                                    expected_indices)

        # Stopping
        data_loader.stop()

    def test_run(self):
        data_loader = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                      shape_out=(3000,),
                                      seed_sub_sampling=0)

        # Run parameters
        num_steps = 9
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        # Running
        data_loader.run(condition=run_cnd, run_cfg=run_cfg)

        # Stopping
        data_loader.stop()

        self.assertFalse(data_loader.runtime._is_running)

    def test_end_of_file(self):
        data_loader = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                      shape_out=(3000,),
                                      seed_sub_sampling=0)

        # Run parameters
        num_steps = 10
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=num_steps)

        # Running
        data_loader.run(condition=run_cnd, run_cfg=run_cfg)

        # Stopping
        data_loader.stop()

        self.assertFalse(data_loader.runtime._is_running)

    def test_index_encoding(self):
        x_history = [
            [8, 12, 13, 13, 13, 9, 14, 14, 13, 13, 8, 9, 9, 13, 9],
            [8, 9, 12, 7, 12, 12, 20, 19, 10],
            [39, 12, 13, 8, 16, 7, 7, 7, 7, 16, 7, 12, 7, 7, 12],
            [10, 7, 10, 8, 8, 7],
            [2],
            [12, 10, 7],
            [22],
            [9],
            [21]
        ]
        y_history = [
            [157, 148, 146, 156, 158, 167, 122, 113, 149, 148, 156,
             109, 107, 160, 160],
            [160, 112, 137, 128, 130, 145, 104, 99, 111],
            [118, 141, 131, 161, 102, 104, 119, 126, 124, 103, 130,
             129, 141, 102, 133],
            [110, 122, 109, 122, 166, 121],
            [104],
            [163, 108, 133],
            [102],
            [172],
            [109]
        ]
        seed_rng = 0
        rng = np.random.default_rng(seed=seed_rng)
        dense_shape = (240, 180)

        max_num_events = 10
        data_loader = AedatDataLoader(file_path="../dvs_recording.aedat4",
                                      shape_out=(max_num_events,),
                                      seed_sub_sampling=seed_rng)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        # Run parameters
        num_steps = 9
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        for i in range(num_steps):
            data_loader.run(condition=run_cnd, run_cfg=run_cfg)

            expected_xs = np.array(x_history[i])
            expected_ys = np.array(y_history[i])

            sent_and_received_indices = \
                recv_sparse.idx.get()[:expected_xs.shape[0]].astype(int)

            reconstructed_xs, reconstructed_ys = \
                np.unravel_index(sent_and_received_indices, dense_shape)

            if expected_xs.shape[0] > max_num_events:
                data_idx_array = np.arange(0, expected_xs.shape[0])
                sampled_idx = rng.choice(data_idx_array,
                                         max_num_events,
                                         replace=False)

                expected_xs = expected_xs[sampled_idx]
                expected_ys = expected_ys[sampled_idx]

            np.testing.assert_equal(reconstructed_xs, expected_xs)
            np.testing.assert_equal(reconstructed_ys, expected_ys)

        # Stopping
        data_loader.stop()


if __name__ == '__main__':
    unittest.main()
