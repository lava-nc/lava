# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from lava.proc.s4d.process import SigmaS4Delta, SigmaS4DeltaLayer
import numpy as np


class TestSigmaS4DeltaProcess(unittest.TestCase):
    """Tests for SigmaS4Delta Class"""

    def test_init(self) -> None:
        """Tests instantiation of SigmaS4Delta"""
        shape = 10
        vth = 10
        state_exp = 6
        s4_exp = 12
        A = np.ones(shape) * 0.5
        B = np.ones(shape) * 0.8
        C = np.ones(shape) * 0.9
        sigmas4delta = SigmaS4Delta(shape=(shape,),
                                    vth=vth,
                                    state_exp=state_exp,
                                    S4_exp=s4_exp,
                                    A=A,
                                    B=B,
                                    C=C)

        # determined by user - S4 part
        self.assertEqual(sigmas4delta.shape, (shape,))
        self.assertEqual(sigmas4delta.vth.init, vth * 2 ** state_exp)
        self.assertEqual(sigmas4delta.S4_exp.init, s4_exp)
        np.testing.assert_array_equal(sigmas4delta.A.init, A)
        np.testing.assert_array_equal(sigmas4delta.B.init, B)
        np.testing.assert_array_equal(sigmas4delta.C.init, C)
        self.assertEqual(sigmas4delta.state_exp.init, state_exp)
        self.assertEqual(sigmas4delta.S4state.init, 0)

        # default sigmadelta params - inherited from SigmaDelta class
        self.assertEqual(sigmas4delta.cum_error.init, False)
        self.assertEqual(sigmas4delta.spike_exp.init, 0)
        self.assertEqual(sigmas4delta.bias.init, 0)


class TestSigmaS4DeltaLayer(unittest.TestCase):
    """Tests for SigmaS4DeltaLayer Class"""

    def test_init(self) -> None:
        """Tests instantiation of SigmaS4DeltaLayer """
        shape = 10
        vth = 10
        state_exp = 6
        s4_exp = 12
        d_states = 5
        A = np.ones(shape * d_states) * 0.5
        B = np.ones(shape * d_states) * 0.8
        C = np.ones(shape * d_states) * 0.9

        sigmas4deltalayer = SigmaS4DeltaLayer(shape=(shape,),
                                              d_states=d_states,
                                              vth=vth,
                                              state_exp=state_exp,
                                              S4_exp=s4_exp,
                                              A=A,
                                              B=B,
                                              C=C)
        # determined by user - S4 part
        self.assertEqual(sigmas4deltalayer.shape, (shape,))
        self.assertEqual(sigmas4deltalayer.S4_exp.init, s4_exp)
        np.testing.assert_array_equal(sigmas4deltalayer.A.init, A)
        np.testing.assert_array_equal(sigmas4deltalayer.B.init, B)
        np.testing.assert_array_equal(sigmas4deltalayer.C.init, C)
        self.assertEqual(sigmas4deltalayer.state_exp.init, state_exp)
        self.assertEqual(sigmas4deltalayer.S4state.init, 0)

        # determined by user/via number of states and shape
        np.testing.assert_array_equal(sigmas4deltalayer.conn_weights.init,
                                      np.kron(np.eye(shape), np.ones(d_states)))


if __name__ == '__main__':
    unittest.main()
