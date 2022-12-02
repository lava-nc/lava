# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.event_data.event_pre_processor.utils import DownSamplingMethodDense

from lava.proc.event_data.event_data_loader.aedat_data_loader import AedatDataLoader
from lava.proc.event_data.event_pre_processor.sparse_to_sparse.binary_to_unary_polarity \
    import BinaryToUnaryPolarity
from lava.proc.event_data.event_pre_processor.sparse_to_dense.sparse_to_dense import \
    SparseToDense
from lava.proc.event_data.event_pre_processor.dense_to_dense.down_sampling_dense import \
    DownSamplingDense

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


class TestEventDataIntegration(unittest.TestCase):
    def test_integration(self):
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

        # AedatDataLoader parameters
        adl_file_path = "dvs_recording.aedat4"
        adl_max_num_events_out = 10
        adl_shape_out = (adl_max_num_events_out,)
        # BinaryToUnaryPolarity parameters
        btup_shape = adl_shape_out
        # SparseToDense parameters
        std_shape_in = btup_shape
        std_width_out = 240
        std_height_out = 180
        std_shape_out = (std_width_out, std_height_out)
        # DownSamplingDense parameters
        dss_shape_in = std_shape_out
        dss_down_sampling_method = DownSamplingMethodDense.MAX_POOLING
        dss_down_sampling_factor = 1
        # RecvDense parameters
        rd_shape = (dss_shape_in[0] // dss_down_sampling_factor,
                    dss_shape_in[1] // dss_down_sampling_factor)

        # Instantiating Processes
        aedat_data_loader = AedatDataLoader(file_path=adl_file_path,
                                            shape_out=adl_shape_out,
                                            seed_sub_sampling=seed_rng)
        binary_to_unary_polarity = BinaryToUnaryPolarity(shape=btup_shape)
        sparse_to_dense = SparseToDense(shape_in=std_shape_in,
                                        shape_out=std_shape_out)
        down_sampling_dense = DownSamplingDense(
            shape_in=dss_shape_in,
            down_sampling_method=dss_down_sampling_method,
            down_sampling_factor=dss_down_sampling_factor
        )
        recv_dense = RecvDense(shape=rd_shape)

        # Connecting Processes
        aedat_data_loader.out_port.connect(binary_to_unary_polarity.in_port)
        binary_to_unary_polarity.out_port.connect(sparse_to_dense.in_port)
        sparse_to_dense.out_port.connect(down_sampling_dense.in_port)
        down_sampling_dense.out_port.connect(recv_dense.in_port)

        # Run parameters
        num_steps = 9
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        for i in range(num_steps):
            aedat_data_loader.run(condition=run_cnd, run_cfg=run_cfg)

            xs = np.array(x_history[i])
            ys = np.array(y_history[i])

            sent_and_received_data = \
                recv_dense.data.get().astype(int)

            if xs.shape[0] > adl_max_num_events_out:
                data_idx_array = np.arange(0, xs.shape[0])
                sampled_idx = rng.choice(data_idx_array,
                                         adl_max_num_events_out,
                                         replace=False)

                xs = xs[sampled_idx]
                ys = ys[sampled_idx]

            expected_data = np.zeros(std_shape_out)
            expected_data[xs, ys] = 1

            np.testing.assert_equal(sent_and_received_data, expected_data)

        # Stopping
        aedat_data_loader.stop()


if __name__ == '__main__':
    unittest.main()
