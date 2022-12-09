# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import unittest
from pathlib import Path

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.io.aedat_stream import AedatStream, AedatStreamPM

from ..utils import RecvSparse

current_directory = Path(__file__).resolve().parent
aedat_file_path = str(current_directory.parent / "dvs_recording.aedat4")


class TestProcessAedatStream(unittest.TestCase):
    def test_init(self) -> None:
        """Tests instantiation of AedatStream."""
        data_loader = AedatStream(file_path=aedat_file_path,
                                  shape_out=(43200,))

        self.assertIsInstance(data_loader, AedatStream)
        self.assertEqual(data_loader.proc_params["file_path"], aedat_file_path)
        self.assertEqual(data_loader.out_port.shape, (43200,))

    def test_unsupported_file_extension_throws_exception(self) -> None:
        """Tests whether a file_path argument with an unsupported file
        extension throws an exception."""
        unsupported_extension = "py"
        with(self.assertRaises(ValueError)):
            AedatStream(file_path="test_file." + unsupported_extension,
                        shape_out=(43200,))

    def test_missing_file_throws_exception(self) -> None:
        """Tests whether an exception is thrown when a specified file does not
        exist."""
        with(self.assertRaises(FileNotFoundError)):
            AedatStream(file_path="missing_file.aedat4",
                        shape_out=(43200,))

    def test_invalid_shape_throws_exception(self) -> None:
        """Tests whether an invalid shape_out (not one-dimensional)
        throws an exception."""
        invalid_shape = (240, 180)
        with(self.assertRaises(ValueError)):
            AedatStream(file_path=aedat_file_path,
                        shape_out=invalid_shape)

    def test_negative_size_throws_exception(self) -> None:
        """Tests whether shape_out with a negative size throws an exception."""
        invalid_size = -43200
        with(self.assertRaises(ValueError)):
            AedatStream(file_path=aedat_file_path,
                        shape_out=(invalid_size,))


class TestProcessModelAedatStream(unittest.TestCase):
    def test_init(self):
        """Tests instantiation of the AedatStream ProcessModel."""
        proc_params = {
            "file_path": aedat_file_path,
            "shape_out": (3000,),
            "seed_sub_sampling": 0
        }
        pm = AedatStreamPM(proc_params)
        self.assertIsInstance(pm, AedatStreamPM)

    def test_streaming_from_aedat_file(self):
        """Tests streaming from an aedat file."""
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

        max_num_events = 15
        data_loader = AedatStream(file_path=aedat_file_path,
                                  shape_out=(max_num_events,))
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        num_steps = 5
        for i in range(num_steps):
            data_loader.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg())

            expected_data = np.array(data_history[i])
            expected_indices = np.array(indices_history[i])

            received_data = recv_sparse.data.get()[:expected_data.shape[0]]
            received_indices = recv_sparse.idx.get()[:expected_indices.shape[0]]

            np.testing.assert_equal(received_data, expected_data)
            np.testing.assert_equal(received_indices, expected_indices)

        data_loader.stop()

    def test_streaming_from_aedat_file_with_sub_sampling(self):
        """Tests streaming from an aedat file when sub-sampling of the stream
        becomes necessary. This is the case when the max_num_events is
        smaller than the amount of events we receive in a given batch.
        """
        expected_data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0]
        ]
        expected_indices = [
            [1597, 2486, 2496, 2498, 1787, 2633, 1729, 1727, 2500, 1780],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138, 2301, 2982, 1364, 1379, 1386, 1384, 1390, 2289, 1362],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]

        max_num_events = 10
        data_loader = AedatStream(file_path=aedat_file_path,
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=0)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        num_steps = 5
        for i in range(num_steps):
            data_loader.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg())

            received_data = recv_sparse.data.get()[:len(expected_data[i])]
            received_indices = recv_sparse.idx.get()[:len(expected_indices[i])]

            np.testing.assert_equal(received_data, expected_data[i])
            np.testing.assert_equal(received_indices, expected_indices[i])

        data_loader.stop()

    def test_randomness_of_sub_sampling(self):
        """Tests whether sub-sampling uses a random component to select the
        events that are discarded. Using different seeds should result in
        different sub-sampling."""
        expected_indices_seed_0 = [
            [1597, 2486, 2496, 2498, 1787, 2633, 1729, 1727, 2500, 1780],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138, 2301, 2982, 1364, 1379, 1386, 1384, 1390, 2289, 1362],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]
        expected_indices_seed_1 = [
            [1597, 2308, 2486, 2496, 2498, 2642, 2489, 2488, 1727, 2500],
            [1600, 1732, 2297, 1388, 2290, 2305, 3704, 3519, 1911],
            [7138, 2301, 1601, 1364, 1379, 1386, 1384, 1390, 2289, 1401],
            [1910, 1382, 1909, 1562, 1606, 1381],
            [464]
        ]
        received_indices_0 = []
        received_indices_1 = []

        max_num_events = 10

        # Architecture with seed 0.
        data_loader_0 = AedatStream(file_path=aedat_file_path,
                                    shape_out=(max_num_events,),
                                    seed_sub_sampling=0)
        recv_sparse_0 = RecvSparse(shape=(max_num_events,))
        data_loader_0.out_port.connect(recv_sparse_0.in_port)

        # Architecture with seed 1.
        data_loader_1 = AedatStream(file_path=aedat_file_path,
                                    shape_out=(max_num_events,),
                                    seed_sub_sampling=1)
        recv_sparse_1 = RecvSparse(shape=(max_num_events,))
        data_loader_1.out_port.connect(recv_sparse_1.in_port)

        num_steps = 5
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Run architecture with seed 0.
        for i in range(num_steps):
            data_loader_0.run(condition=run_cnd, run_cfg=Loihi1SimCfg())
            idx = self._extract_idx(recv_sparse_0, expected_indices_seed_0, i)
            received_indices_0.append(idx)
        data_loader_0.stop()

        # Run architecture with seed 1.
        for i in range(num_steps):
            data_loader_1.run(condition=run_cnd, run_cfg=run_cfg)
            idx = self._extract_idx(recv_sparse_1, expected_indices_seed_1, i)
            received_indices_1.append(idx)
        data_loader_1.stop()

        # Indices from the individual runs should be as expected.
        np.testing.assert_equal(received_indices_0, expected_indices_seed_0)
        np.testing.assert_equal(received_indices_1, expected_indices_seed_1)
        # Indices from the two runs should be different.
        self.assertTrue(np.any(received_indices_0 != received_indices_1))

    @staticmethod
    def _extract_idx(recv_sparse: RecvSparse,
                     expected: ty.List[ty.List[int]],
                     time_step: int) -> ty.List[ty.List[int]]:
        idx_array = recv_sparse.idx.get().astype(int)
        idx_array_cropped = idx_array[:len(expected[time_step])]
        return list(idx_array_cropped)

    def test_looping_over_end_of_file(self):
        """Tests whether the stream loops back to the beginning of the aedat
        file when reaching the end of the file.
        """
        max_num_events = 15
        data_loader = AedatStream(file_path=aedat_file_path,
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=0)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        received_data = []
        received_indices = []

        num_steps = 32
        for i in range(num_steps):
            data_loader.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg())
            received_data.append(recv_sparse.data.get())
            received_indices.append(recv_sparse.idx.get())
        data_loader.stop()

        #  The test file contains 27 time-steps. It is expected that the
        #  stream returns to the first entry in the 28th time step.
        np.testing.assert_equal(received_data[27], received_data[0])
        np.testing.assert_equal(received_indices[27], received_indices[0])

    def test_index_encoding(self):
        """Tests whether indices are correctly converted from (x,y) format to
        a linear index."""
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
        dense_shape = (240, 180)

        max_num_events = 15
        data_loader = AedatStream(file_path=aedat_file_path,
                                  shape_out=(max_num_events,),
                                  seed_sub_sampling=0)
        recv_sparse = RecvSparse(shape=(max_num_events,))

        data_loader.out_port.connect(recv_sparse.in_port)

        num_steps = 9
        for i in range(num_steps):
            data_loader.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg())

            expected_xs = np.array(x_history[i])
            expected_ys = np.array(y_history[i])

            received_indices = \
                recv_sparse.idx.get()[:expected_xs.shape[0]].astype(int)

            reconstructed_xs, reconstructed_ys = \
                np.unravel_index(received_indices, dense_shape)

            np.testing.assert_equal(reconstructed_xs, expected_xs)
            np.testing.assert_equal(reconstructed_ys, expected_ys)

        data_loader.stop()


if __name__ == '__main__':
    unittest.main()
