# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.event_data.io.aedat_stream import AedatStream
from lava.proc.event_data.binary_to_unary_polarity.process \
    import BinaryToUnaryPolarity
from lava.proc.event_data.events_to_frame.process import EventsToFrame


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
    def test_integration_aedat_stream(self):
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

        # AedatStream parameters
        as_file_path = "dvs_recording.aedat4"
        as_max_num_events_out = 10
        as_shape_out = (as_max_num_events_out,)
        # BinaryToUnaryPolarity parameters
        btup_shape = as_shape_out
        # EventsToFrame parameters
        etf_shape_in = btup_shape
        etf_width_out = 240
        etf_height_out = 180
        etf_num_channels = 1
        etf_shape_out = (etf_width_out, etf_height_out, etf_num_channels)
        # RecvDense parameters
        rd_shape = etf_shape_out

        # Instantiating Processes
        aedat_stream = AedatStream(file_path=as_file_path,
                                   shape_out=as_shape_out,
                                   seed_sub_sampling=seed_rng)
        binary_to_unary_polarity = BinaryToUnaryPolarity(shape=btup_shape)
        events_to_frame = EventsToFrame(shape_in=etf_shape_in,
                                        shape_out=etf_shape_out)
        recv_dense = RecvDense(shape=rd_shape)

        # Connecting Processes
        aedat_stream.out_port.connect(binary_to_unary_polarity.in_port)
        binary_to_unary_polarity.out_port.connect(events_to_frame.in_port)
        events_to_frame.out_port.connect(recv_dense.in_port)

        # Run parameters
        num_steps = 9
        run_cfg = Loihi1SimCfg()
        run_cnd = RunSteps(num_steps=1)

        # Running
        for i in range(num_steps):
            aedat_stream.run(condition=run_cnd, run_cfg=run_cfg)

            xs = np.array(x_history[i])
            ys = np.array(y_history[i])

            sent_and_received_data = \
                recv_dense.data.get().astype(int)

            if xs.shape[0] > as_max_num_events_out:
                data_idx_array = np.arange(0, xs.shape[0])
                sampled_idx = rng.choice(data_idx_array,
                                         as_max_num_events_out,
                                         replace=False)

                xs = xs[sampled_idx]
                ys = ys[sampled_idx]

            expected_data = np.zeros(etf_shape_out)
            expected_data[xs, ys] = 1

            np.testing.assert_equal(sent_and_received_data, expected_data)

        # Stopping
        aedat_stream.stop()


if __name__ == '__main__':
    unittest.main()
