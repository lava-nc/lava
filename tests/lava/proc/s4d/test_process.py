# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.s4d.process import SigmaS4dDelta, SigmaS4dDeltaLayer, S4d


class TestS4dProcess(unittest.TestCase):
    """Tests for S4d Class"""

    def test_init(self) -> None:
        """Tests instantiation of S4d"""
        shape = 10
        s4_exp = 12
        inp_exp = 8
        a = np.ones(shape) * 0.5
        b = np.ones(shape) * 0.8
        c = np.ones(shape) * 0.9
        s4d = S4d(shape=(shape,),
                  s4_exp=s4_exp,
                  inp_exp=inp_exp,
                  a=a,
                  b=b,
                  c=c)

        self.assertEqual(s4d.shape, (shape,))
        self.assertEqual(s4d.s4_exp.init, s4_exp)
        self.assertEqual(s4d.inp_exp.init, inp_exp)
        np.testing.assert_array_equal(s4d.a.init, a)
        np.testing.assert_array_equal(s4d.b.init, b)
        np.testing.assert_array_equal(s4d.c.init, c)
        self.assertEqual(s4d.s4_state.init, 0)


class TestSigmaS4dDeltaProcess(unittest.TestCase):
    """Tests for SigmaS4dDelta Class"""

    def test_init(self) -> None:
        """Tests instantiation of SigmaS4dDelta"""
        shape = 10
        vth = 10
        state_exp = 6
        s4_exp = 12
        a = np.ones(shape) * 0.5
        b = np.ones(shape) * 0.8
        c = np.ones(shape) * 0.9
        sigma_s4_delta = SigmaS4dDelta(shape=(shape,),
                                       vth=vth,
                                       state_exp=state_exp,
                                       s4_exp=s4_exp,
                                       a=a,
                                       b=b,
                                       c=c)

        # determined by user - S4 part
        self.assertEqual(sigma_s4_delta.shape, (shape,))
        self.assertEqual(sigma_s4_delta .vth.init, vth * 2 ** state_exp)
        self.assertEqual(sigma_s4_delta.s4_exp.init, s4_exp)
        np.testing.assert_array_equal(sigma_s4_delta.a.init, a)
        np.testing.assert_array_equal(sigma_s4_delta.b.init, b)
        np.testing.assert_array_equal(sigma_s4_delta.c.init, c)
        self.assertEqual(sigma_s4_delta.state_exp.init, state_exp)
        self.assertEqual(sigma_s4_delta.s4_state.init, 0)

        # default sigma-delta params - inherited from SigmaDelta class
        self.assertEqual(sigma_s4_delta.cum_error.init, False)
        self.assertEqual(sigma_s4_delta.spike_exp.init, 0)
        self.assertEqual(sigma_s4_delta.bias.init, 0)


class TestSigmaS4DeltaLayer(unittest.TestCase):
    """Tests for SigmaS4dDeltaLayer Class"""

    def test_init(self) -> None:
        """Tests instantiation of SigmaS4dDeltaLayer """
        shape = 10
        vth = 10
        state_exp = 6
        s4_exp = 12
        d_states = 5
        a = np.ones(shape) * 0.5
        b = np.ones(shape) * 0.8
        c = np.ones(shape) * 0.9

        sigma_s4d_delta_layer = SigmaS4dDeltaLayer(shape=(shape,),
                                                   d_states=d_states,
                                                   vth=vth,
                                                   state_exp=state_exp,
                                                   s4_exp=s4_exp,
                                                   a=a,
                                                   b=b,
                                                   c=c)
        # determined by user - S4 part
        self.assertEqual(sigma_s4d_delta_layer.shape, (shape,))
        self.assertEqual(sigma_s4d_delta_layer.S4_exp.init, s4_exp)
        np.testing.assert_array_equal(sigma_s4d_delta_layer.a.init, a)
        np.testing.assert_array_equal(sigma_s4d_delta_layer.b.init, b)
        np.testing.assert_array_equal(sigma_s4d_delta_layer.c.init, c)
        self.assertEqual(sigma_s4d_delta_layer.state_exp.init, state_exp)
        self.assertEqual(sigma_s4d_delta_layer.s4_state.init, 0)

        # determined by user/via number of states and shape
        np.testing.assert_array_equal(sigma_s4d_delta_layer.conn_weights.init,
                                      np.kron(np.eye(shape), np.ones(d_states)))


if __name__ == '__main__':
    unittest.main()
