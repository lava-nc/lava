# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.proc.prodneuron.process import ProdNeuron
from lava.proc.dense.process import Dense
from lava.proc import io

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


class TestProdNeuronProc(unittest.TestCase):
    """Tests for ProdNeuron."""

    def test_prod_neuron_out(self):
        """Tests prod neuron calcultion is correct."""
        weight_exp = 7
        num_steps = 10

        weights1 = np.zeros((1, 1))
        weights1[0, 0] = 1

        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        weights2 = np.zeros((1, 1))
        weights2[0, 0] = 0.5

        weights2 *= 2**weight_exp
        weights2 = weights2.astype('int')

        inp_data1 = np.zeros((weights1.shape[1], num_steps))
        inp_data2 = np.zeros((weights2.shape[1], num_steps))

        inp_data1[:, 2] = 10
        inp_data1[:, 6] = 30
        inp_data2[:, :] = 20

        dense1 = Dense(weights=weights1,
                       num_message_bits=24,
                       weight_exp=-weight_exp)
        dense2 = Dense(weights=weights2,
                       num_message_bits=24,
                       weight_exp=-weight_exp)

        vec1 = ProdNeuron(shape=(weights1.shape[0],))

        generator1 = io.source.RingBuffer(data=inp_data1)
        generator2 = io.source.RingBuffer(data=inp_data2)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator1.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in1)

        generator2.s_out.connect(dense2.s_in)
        dense2.a_out.connect(vec1.a_in2)

        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ch1 = (weights1 @ inp_data1) / 2**weight_exp
        ch2 = (weights2 @ inp_data2) / 2**weight_exp

        expected_out = ch1 * ch2
        # Then there is one extra timestep from hardware
        self.assertTrue(np.all(expected_out[:, :-1] == out_data[:, 1:]))


if __name__ == '__main__':
    unittest.main()
