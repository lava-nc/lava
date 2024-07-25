# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from lava.proc.graded.process import (GradedVec, GradedReluVec,
                                      NormVecDelay, InvSqrt)
from lava.proc.graded.models import inv_sqrt
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.proc import io

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


class TestGradedVecProc(unittest.TestCase):
    """Tests for GradedVec."""

    def test_gradedvec_dot_dense(self):
        """Tests that GradedVec and Dense computes dot product."""
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10, 1))
        weights1[:, 0] = (np.arange(10) - 5) * 0.2

        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000

        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        dense1 = Dense(weights=weights1, num_message_bits=24,
                       weight_exp=-weight_exp)
        vec1 = GradedVec(shape=(weights1.shape[0],),
                         vth=v_thresh)

        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ww = np.floor(weights1 / 2) * 2
        expected_out = np.floor((ww @ inp_data) / 2**weight_exp)

        self.assertTrue(np.all(out_data[:, (3, 7)] == expected_out[:, (2, 6)]))

    def test_gradedvec_dot_sparse(self):
        """Tests that GradedVec and Sparse computes dot product"""
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10, 1))
        weights1[:, 0] = (np.arange(10) - 5) * 0.2

        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000

        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        sparse1 = Sparse(weights=csr_matrix(weights1),
                         num_message_bits=24,
                         weight_exp=-weight_exp)
        vec1 = GradedVec(shape=(weights1.shape[0],),
                         vth=v_thresh)

        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator.s_out.connect(sparse1.s_in)
        sparse1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ww = np.floor(weights1 / 2) * 2
        expected_out = np.floor((ww @ inp_data) / 2**weight_exp)

        self.assertTrue(np.all(out_data[:, (3, 7)] == expected_out[:, (2, 6)]))


class TestGradedReluVecProc(unittest.TestCase):
    """Tests for GradedReluVec"""

    def test_gradedreluvec_dot_dense(self):
        """Tests that GradedReluVec and Dense computes dot product"""
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10, 1))
        weights1[:, 0] = (np.arange(10) - 5) * 0.2

        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000

        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        dense1 = Dense(weights=weights1, num_message_bits=24,
                       weight_exp=-weight_exp)
        vec1 = GradedReluVec(shape=(weights1.shape[0],),
                             vth=v_thresh)

        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ww = np.floor(weights1 / 2) * 2
        expected_out = np.floor((ww @ inp_data) / 2**weight_exp)
        expected_out *= expected_out > v_thresh

        self.assertTrue(np.all(out_data[:, (3, 7)] == expected_out[:, (2, 6)]))

    def test_gradedreluvec_dot_sparse(self):
        """Tests that GradedReluVec and Sparse computes dot product"""
        num_steps = 10
        v_thresh = 1

        weights1 = np.zeros((10, 1))
        weights1[:, 0] = (np.arange(10) - 5) * 0.2

        inp_data = np.zeros((weights1.shape[1], num_steps))
        inp_data[:, 2] = 1000
        inp_data[:, 6] = 20000

        weight_exp = 7
        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        sparse1 = Sparse(weights=csr_matrix(weights1),
                         num_message_bits=24,
                         weight_exp=-weight_exp)
        vec1 = GradedReluVec(shape=(weights1.shape[0],),
                             vth=v_thresh)

        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator.s_out.connect(sparse1.s_in)
        sparse1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ww = np.floor(weights1 / 2) * 2
        expected_out = np.floor((ww @ inp_data) / 2**weight_exp)
        expected_out *= expected_out > v_thresh

        self.assertTrue(np.all(out_data[:, (3, 7)] == expected_out[:, (2, 6)]))


class TestInvSqrtProc(unittest.TestCase):
    """Tests for inverse square process."""

    def test_invsqrt_calc(self):
        """Checks the InvSqrt calculation."""
        fp_base = 12  # Base of the decimal point

        num_steps = 25
        weights1 = np.zeros((1, 1))
        weights1[0, 0] = 1
        weight_exp = 7

        weights1 *= 2**weight_exp
        weights1 = weights1.astype('int')

        in_vals = [2**(i % 24) for i in range(num_steps)]

        inp_data = np.zeros((weights1.shape[1], num_steps))

        for i in range(num_steps):
            inp_data[:, i] = in_vals[i]

        dense1 = Dense(weights=weights1,
                       num_message_bits=24,
                       weight_exp=-weight_exp)
        vec1 = InvSqrt(shape=(weights1.shape[0],),
                       fp_base=fp_base)

        generator = io.source.RingBuffer(data=inp_data)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in)
        vec1.s_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        expected_out = np.array([inv_sqrt(inp_data[0, i], 5)
                                 for i in range(num_steps)])

        self.assertTrue(np.all(expected_out[:-1] == out_data[:, 1:]))


class TestNormVecDelayProc(unittest.TestCase):
    """Tests for NormVecDelay."""

    def test_norm_vec_delay_out1(self):
        """Checks the first channel output of NormVecDelay."""
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

        vec1 = NormVecDelay(shape=(weights1.shape[0],))

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

        # I'm using roll to account for the two step delay in NormVecDelay.
        # However, this is a hack, as the inputs need to be 0 at the end
        # of the simulation, since roll wraps the values.
        # Be wary that this potentially won't be correct with different inputs.
        # There seems to be a delay step missing compared to
        # ncmodel, not sure where the delay should go...
        expected_out = np.roll(ch1, 1) * ch2

        # Then there is one extra delay timestep from hardware
        self.assertTrue(np.all(expected_out[:, :-1] == out_data[:, 1:]))

    def test_norm_vec_delay_out2(self):
        """Checks the second channel output of NormVecDelay."""
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

        vec1 = NormVecDelay(shape=(weights1.shape[0],))

        generator1 = io.source.RingBuffer(data=inp_data1)
        generator2 = io.source.RingBuffer(data=inp_data2)
        logger = io.sink.RingBuffer(shape=(weights1.shape[0],),
                                    buffer=num_steps)

        generator1.s_out.connect(dense1.s_in)
        dense1.a_out.connect(vec1.a_in1)

        generator2.s_out.connect(dense2.s_in)
        dense2.a_out.connect(vec1.a_in2)

        vec1.s2_out.connect(logger.a_in)

        vec1.run(condition=RunSteps(num_steps=num_steps),
                 run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        out_data = logger.data.get().astype('int')
        vec1.stop()

        ch1 = (weights1 @ inp_data1) / 2**weight_exp
        expected_out = ch1 ** 2
        # Then there is one extra timestep from hardware
        self.assertTrue(np.all(expected_out[:, :-1] == out_data[:, 1:]))


if __name__ == '__main__':
    unittest.main()
