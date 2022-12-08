# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from dv import AedatFile
from dv.AedatFile import _AedatFileEventNumpyPacketIterator
import numpy as np
import typing as ty
import unittest

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
from lava.proc.event_data.io.aedat_stream import AedatStream, AedatStreamPM


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


class TestProcessAedatStream(unittest.TestCase):
    def test_init(self):
        """
        Tests instantiation of AedatStream.
        """
        data_loader = AedatStream(file_path="../dvs_recording.aedat4",
                                  shape_out=(43200,))

        self.assertIsInstance(data_loader, AedatStream)
        self.assertEqual(data_loader.proc_params["file_path"],
                         "../dvs_recording.aedat4")
        self.assertEqual(data_loader.proc_params["shape_out"], (43200,))

    def test_unsupported_file_extension_throws_exception(self):
        """
        Tests whether a file_path argument with an unsupported file extension
        throws an exception.
        """
        with(self.assertRaises(ValueError)):
            AedatStream(file_path="test_aedat_data_loader.py",
                        shape_out=(43200,))

    def test_missing_file_throws_exception(self):
        """
        Tests whether an exception is thrown when a specified file does not exist.
        """
        with(self.assertRaises(FileNotFoundError)):
            AedatStream(file_path="missing_file.aedat4",
                        shape_out=(43200,))

    def test_invalid_shape_throws_exception(self):
        """
        Tests whether a shape_out argument with an invalid shape throws an exception.
        """
        with(self.assertRaises(ValueError)):
            AedatStream(file_path="../dvs_recording.aedat4",
                        shape_out=(240, 180))

    def test_negative_size_throws_exception(self):
        """
        Tests whether a shape_out argument with a negative size throws an exception.
        """
        with(self.assertRaises(ValueError)):
            AedatStream(file_path="../dvs_recording.aedat4",
                        shape_out=(-43200,))


# TODO: add doc strings
class TestProcessModelAedatStream(unittest.TestCase):
    def test_init(self):
        """
        Tests instantiation of the AedatStream process model.
        """
        proc_params = {
            "file_path": "../dvs_recording.aedat4",
            "shape_out": (3000,),
            "seed_sub_sampling": 0
        }

        pm = AedatStreamPM(proc_params)

        self.assertIsInstance(pm, AedatStreamPM)
        self.assertEqual(pm._shape_out, proc_params["shape_out"])
        self.assertIsInstance(pm._file, AedatFile)
        self.assertIsInstance(pm._stream,
                              _AedatFileEventNumpyPacketIterator)
        self.assertIsInstance(pm._frame_shape, tuple)

    def test_run_without_sub_sampling(self):
        """
        Tests whether running yields the expected behavior, given that the
        user parameters are all correct.
        """
        data_history = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0]
        ]
        indices_history = [
            [1597, 2308, 2486, 2496, 2498, 1787, 2642, 2633, 2489,
             2488, 1596, 1729, 1727, 2500, 1780],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138, 2301, 2471, 1601, 2982, 1364, 1379, 1386, 1384,
             2983, 1390, 2289, 1401, 1362, 2293],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]

        seed_rng = 0
        max_num_events = 15
        data_loader = AedatStream(file_path="../dvs_recording.aedat4",
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=seed_rng)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        num_steps = 5
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            data_loader.run(condition=run_cnd, run_cfg=run_cfg)

            expected_data = np.array(data_history[i])
            expected_indices = np.array(indices_history[i])

            sent_and_received_data = \
                recv_sparse.data.get()[:expected_data.shape[0]]
            sent_and_received_indices = \
                recv_sparse.idx.get()[:expected_indices.shape[0]]

            np.testing.assert_equal(sent_and_received_data,
                                    expected_data)
            np.testing.assert_equal(sent_and_received_indices,
                                    expected_indices)

        data_loader.stop()

    def test_sub_sampling(self):
        """
        Tests whether we get the expected behavior when we set a max_num_events
        that is smaller than the amount of events we receive in a given batch
        (i.e. the process will sub-sample correctly).
        """
        expected_data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0]
        ]

        expected_indices = [
            [1597., 2486., 2496., 2498., 1787., 2633., 1729., 1727., 2500., 1780.],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138., 2301., 2982., 1364., 1379., 1386., 1384., 1390., 2289., 1362.],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]

        seed_rng = 0
        max_num_events = 10
        data_loader = AedatStream(file_path="../dvs_recording.aedat4",
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=seed_rng)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        num_steps = 5
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            data_loader.run(condition=run_cnd, run_cfg=run_cfg)

            sent_and_received_data = \
                recv_sparse.data.get()[:len(expected_data[i])]
            sent_and_received_indices = \
                recv_sparse.idx.get()[:len(expected_indices[i])]

            np.testing.assert_equal(sent_and_received_data,
                                    expected_data[i])
            np.testing.assert_equal(sent_and_received_indices,
                                    expected_indices[i])

        data_loader.stop()

    def test_sub_sampling_seed(self):
        """
        Tests whether using different seeds does indeed result in different samples.
        TODO: would testing on only 1 timestep be sufficient?
        """
        expected_indices_seed_0 = [
            [1597., 2486., 2496., 2498., 1787., 2633., 1729., 1727., 2500., 1780.],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138., 2301., 2982., 1364., 1379., 1386., 1384., 1390., 2289., 1362.],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]

        expected_indices_seed_1 = [
            [1597., 2308., 2486., 2496., 2498., 2642., 2489., 2488., 1727., 2500.],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138., 2301., 1601., 1364., 1379., 1386., 1384., 1390., 2289., 1401.],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]
        sent_and_received_indices_1 = []
        sent_and_received_indices_2 = []

        max_num_events = 10
        seed_rng_run_1 = 0
        seed_rng_run_2 = 1

        data_loader_1 = AedatStream(file_path="../dvs_recording.aedat4",
                                    shape_out=(max_num_events,),
                                    seed_sub_sampling=seed_rng_run_1)
        data_loader_2 = AedatStream(file_path="../dvs_recording.aedat4",
                                    shape_out=(max_num_events,),
                                    seed_sub_sampling=seed_rng_run_2)

        recv_sparse_1 = RecvSparse(shape=(max_num_events,))
        recv_sparse_2 = RecvSparse(shape=(max_num_events,))

        data_loader_1.out_port.connect(recv_sparse_1.in_port)
        data_loader_2.out_port.connect(recv_sparse_2.in_port)

        num_steps = 5
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        for i in range(num_steps):
            data_loader_1.run(condition=run_cnd, run_cfg=run_cfg)

            sent_and_received_indices_1.append \
                (recv_sparse_1.idx.get()[:len(expected_indices_seed_1[i])])

        np.testing.assert_equal(sent_and_received_indices_1,
                                expected_indices_seed_0)

        data_loader_1.stop()

        for i in range(num_steps):
            data_loader_2.run(condition=run_cnd, run_cfg=run_cfg)

            sent_and_received_indices_2.append \
                (recv_sparse_2.idx.get()[:len(expected_indices_seed_1[i])])

        np.testing.assert_equal(sent_and_received_indices_2,
                                expected_indices_seed_1)

        data_loader_2.stop()

    def test_end_of_file(self):
        """
        Tests whether we loop back to the beginning of the event stream when we reach
        the end of the aedat4 file. The test file contains 27 time-steps.
        """
        data_time_steps_1_to_5 = []
        data_time_steps_28_to_32 = []
        indices_time_steps_1_to_5 = []
        indices_time_steps_28_to_32 = []

        seed_rng = 0
        max_num_events = 15
        data_loader = AedatStream(file_path="../dvs_recording.aedat4",
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=seed_rng)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        # Run parameters
        num_steps = 32
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        for i in range(num_steps):
            data_loader.run(condition=run_cnd, run_cfg=run_cfg)
            # get data from the first 5 timesteps
            if i in range(5):
                data_time_steps_1_to_5.append \
                    (recv_sparse.data.get())
                indices_time_steps_1_to_5.append \
                    (recv_sparse.idx.get())

            # get data from timesteps 28-32
            if i in range(27, 32):
                data_time_steps_28_to_32.append \
                    (recv_sparse.data.get())
                indices_time_steps_28_to_32.append \
                    (recv_sparse.idx.get())

        np.testing.assert_equal(data_time_steps_1_to_5,
                                data_time_steps_28_to_32)
        np.testing.assert_equal(indices_time_steps_1_to_5,
                                indices_time_steps_28_to_32)

        # Stopping
        data_loader.stop()

    def test_index_encoding(self):
        """
        Tests whether indices are correctly calculated during the process.
        TODO: have less timesteps? maybe 2? (show it works for multiple timesteps with multiple sizes)? no difference in runtime
        """
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

        max_num_events = 15
        data_loader = AedatStream(file_path="../dvs_recording.aedat4",
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

            np.testing.assert_equal(reconstructed_xs, expected_xs)
            np.testing.assert_equal(reconstructed_ys, expected_ys)

        # Stopping
        data_loader.stop()


if __name__ == '__main__':
    unittest.main()
