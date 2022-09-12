# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

import unittest
import numpy as np

from lava.magma.core.learning.random import TraceRandom, ConnVarRandom


class TestTraceRandom(unittest.TestCase):
    def test_init(self) -> None:
        trace_random = TraceRandom()

        self.assertIsInstance(trace_random, TraceRandom)
        self.assertEqual(trace_random.random_trace_decay.dtype, float)
        self.assertEqual(trace_random.random_impulse_addition.dtype, int)

    def test_advance(self) -> None:
        trace_random = TraceRandom()

        random_trace_decay_old = trace_random.random_trace_decay
        random_impulse_addition_old = trace_random.random_impulse_addition

        trace_random.advance()

        self.assertNotEqual(trace_random.random_trace_decay,
                            random_trace_decay_old)
        self.assertNotEqual(trace_random.random_impulse_addition,
                            random_impulse_addition_old)


class TestConnVarRandom(unittest.TestCase):
    def test_init(self) -> None:
        conn_var_random = ConnVarRandom()

        self.assertIsInstance(conn_var_random, ConnVarRandom)
        self.assertEqual(conn_var_random.random_stochastic_round.dtype, float)

    def test_advance(self) -> None:
        conn_var_random = ConnVarRandom()

        random_random_stochastic_round_old = \
            conn_var_random.random_stochastic_round

        conn_var_random.advance()

        self.assertNotEqual(conn_var_random.random_stochastic_round,
                            random_random_stochastic_round_old)


if __name__ == "__main__":
    unittest.main()
