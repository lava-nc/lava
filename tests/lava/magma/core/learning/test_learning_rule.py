# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.learning.product_series import ProductSeries


class TestLoihiLearningRule(unittest.TestCase):
    def test_learning_rule_dw(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertEqual(learning_rule.dd, None)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 1)
        self.assertSetEqual(learning_rule.active_traces, {"x1", "y1"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1"},
            "y0": {"x1"}
        })

    def test_learning_rule_dw_dd(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw and dd, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'x0*y2*w'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 2)
        self.assertSetEqual(learning_rule.active_traces, {"x1", "y1", "y2"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1", "y2"},
            "y0": {"x1"}
        })

    def test_learning_rule_dw_dd_dt(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with string
        learning rule for dw, dd and dt, impulse and tau values for x1 and y1,
        and t_epoch."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'x0*y2*w'
        dt = 'x0*y3*sgn(d) + y0*x2'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd, dt=dt,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertIsInstance(learning_rule.dt, ProductSeries)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, None)
        self.assertEqual(len(learning_rule.active_product_series), 3)
        self.assertSetEqual(learning_rule.active_traces,
                            {"x1", "x2", "y1", "y2", "y3"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1", "y2", "y3"},
            "y0": {"x1", "x2"}
        })

    def test_learning_rule_uk_dependency(self) -> None:
        """Tests that a LoihiLearningRule is instantiable with a string
        learning rule containing a uk dependency."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        dd = 'u0*x2*y2'
        impulse = 16
        tau = 10
        t_epoch = 1

        learning_rule = LoihiLearningRule(dw=dw, dd=dd,
                                          x1_impulse=impulse, x1_tau=tau,
                                          y1_impulse=impulse, y1_tau=tau,
                                          t_epoch=t_epoch)

        self.assertIsInstance(learning_rule, LoihiLearningRule)
        self.assertIsInstance(learning_rule.dw, ProductSeries)
        self.assertIsInstance(learning_rule.dd, ProductSeries)
        self.assertEqual(learning_rule.dt, None)
        self.assertEqual(learning_rule.x1_impulse, impulse)
        self.assertEqual(learning_rule.x1_tau, tau)
        self.assertEqual(learning_rule.y1_impulse, impulse)
        self.assertEqual(learning_rule.y1_tau, tau)
        self.assertEqual(learning_rule.t_epoch, t_epoch)
        self.assertEqual(learning_rule.decimate_exponent, 0)
        self.assertEqual(len(learning_rule.active_product_series), 2)
        self.assertSetEqual(learning_rule.active_traces,
                            {"x1", "x2", "y1", "y2"})
        self.assertDictEqual(learning_rule.active_traces_per_dependency, {
            "x0": {"y1"},
            "y0": {"x1"},
            "u": {"x2", "y2"}
        })

    def test_invalid_impulse(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        impulse is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = -16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_invalid_tau(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        tau is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = -10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_invalid_t_epoch(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        t_epoch is negative."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1'
        impulse = 16
        tau = 10
        t_epoch = -1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_different_decimate_exponent_same_learning_rule(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        providing a learning rule with uk dependencies with different
        decimate exponents."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1 + u0*x2*y2 + u1*y3'
        impulse = 16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)

    def test_different_decimate_exponent_different_learning_rule(self) -> None:
        """Tests that instantiating a LoihiLearningRule throws error when
        providing different learning rules with uk dependencies with different
        decimate exponents."""
        dw = 'x0*(-1)*2^-1*y1 + y0*1*2^1*x1 + u1*y3'
        dd = 'u0*x2*y2'
        impulse = 16
        tau = 10
        t_epoch = 1

        with self.assertRaises(ValueError):
            LoihiLearningRule(dw=dw, dd=dd,
                              x1_impulse=impulse, x1_tau=tau,
                              y1_impulse=impulse, y1_tau=tau,
                              t_epoch=t_epoch)


if __name__ == "__main__":
    unittest.main()
