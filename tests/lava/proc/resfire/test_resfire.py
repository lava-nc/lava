# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.proc.resfire.process import RFZero
from lava.proc.dense.process import Dense
from lava.proc import io

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


class TestRFZeroProc(unittest.TestCase):
    """Tests for RFZero"""

    def test_rfzero_impulse(self):
        """Tests for correct behavior of RFZero neurons from impulse input"""

        num_steps = 50
        num_neurons = 4
        num_inputs = 1

        # Create some weights
        weightr = np.zeros((num_neurons, num_inputs))
        weighti = np.zeros((num_neurons, num_inputs))

        weightr[0, 0] = 50
        weighti[0, 0] = -50

        weightr[1, 0] = -70
        weighti[1, 0] = -70

        weightr[2, 0] = -90
        weighti[2, 0] = 90

        weightr[3, 0] = 110
        weighti[3, 0] = 110

        # Create inputs
        inp_shape = (num_inputs,)
        out_shape = (num_neurons,)

        inp_data = np.zeros((inp_shape[0], num_steps))
        inp_data[:, 3] = 10

        # Create the procs
        denser = Dense(weights=weightr, num_message_bits=24)
        densei = Dense(weights=weighti, num_message_bits=24)

        vec = RFZero(shape=out_shape, uth=1,
                     decay_tau=0.1, freqs=20)

        generator1 = io.source.RingBuffer(data=inp_data)
        generator2 = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=out_shape, buffer=num_steps)

        # Connect the procs
        generator1.s_out.connect(denser.s_in)
        generator2.s_out.connect(densei.s_in)

        denser.a_out.connect(vec.u_in)
        densei.a_out.connect(vec.v_in)

        vec.s_out.connect(logger.a_in)

        # Run
        try:
            vec.run(condition=RunSteps(num_steps=num_steps),
                    run_cfg=Loihi2SimCfg())
            out_data = logger.data.get().astype(np.int32)
        finally:
            vec.stop()

        expected_out = np.array([661, 833, 932, 1007])

        self.assertTrue(
            np.all(expected_out == out_data[[0, 1, 2, 3], [11, 23, 36, 48]]))


if __name__ == '__main__':
    unittest.main()
