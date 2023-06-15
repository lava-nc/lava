# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.learning.random import TraceRandom, ConnVarRandom


class TestTraceRandom(unittest.TestCase):
    def test_init(self) -> None:
        """Tests creating a valid TraceRandom object."""
        trace_random = TraceRandom()

        self.assertIsInstance(trace_random, TraceRandom)
        self.assertEqual(trace_random.random_trace_decay.dtype, float)
        self.assertEqual(trace_random.random_impulse_addition.dtype, int)

    def test_advance(self) -> None:
        """Tests advance on TraceRandom object."""
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
        """Tests creating a valid ConnVarRandom object."""
        conn_var_random = ConnVarRandom()

        self.assertIsInstance(conn_var_random, ConnVarRandom)
        self.assertEqual(conn_var_random.random_stochastic_round.dtype, float)

    def test_advance(self) -> None:
        """Tests advance on ConnVarRandom object."""
        conn_var_random = ConnVarRandom()

        random_random_stochastic_round_old = \
            conn_var_random.random_stochastic_round

        conn_var_random.advance()

        self.assertNotEqual(conn_var_random.random_stochastic_round,
                            random_random_stochastic_round_old)


if __name__ == "__main__":
    unittest.main()
