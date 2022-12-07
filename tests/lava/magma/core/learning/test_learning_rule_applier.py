# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.learning.symbolic_equation import SymbolicEquation
from lava.magma.core.learning.product_series import ProductSeries
from lava.magma.core.learning.learning_rule_applier import \
    LearningRuleApplierFloat, LearningRuleApplierBitApprox


class TestLearningRuleFloatApplier(unittest.TestCase):
    def test_learning_rule_float_applier(self) -> None:
        """Known value test for LearningRuleApplierFloat's apply method."""
        default_rng = np.random.default_rng(seed=0)

        target = "dw"
        str_learning_rule = \
            'x0*(-1)*2^-1*y1 + y0*1*2^1*x1*w + u0*x2 + u0*sgn(w+4)'

        u = 1

        x0 = np.array([0, 0, 1, 1, 0])
        x_shape = x0.shape
        x1_y0 = default_rng.random(x_shape) * 16
        x2_u = default_rng.random(x_shape) * 16

        y0 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1])
        y_shape = y0.shape
        y1_x0 = default_rng.random(y_shape) * 16

        conn_shape = (y0.shape[0], x0.shape[0])
        w = default_rng.random(conn_shape)

        traces = np.zeros((3, 5) + conn_shape)
        traces[0, 2, :, :] = np.broadcast_to(y1_x0[:, np.newaxis], conn_shape)
        traces[1, 0, :, :] = np.broadcast_to(x1_y0[np.newaxis, :], conn_shape)
        traces[2, 1, :, :] = np.broadcast_to(x2_u[np.newaxis, :], conn_shape)

        se = SymbolicEquation(target, str_learning_rule)
        product_series = ProductSeries(se)

        learning_rule_applier = LearningRuleApplierFloat(product_series)

        self.assertIsInstance(learning_rule_applier, LearningRuleApplierFloat)

        # Shape x0: (num_pre_neurons) -> (1, num_pre_neurons)
        # Shape y0: (num_post_neurons) -> (num_post_neurons, 1)
        # Shape w: (num_post_neurons, num_pre_neurons)
        # Shape traces : (3, 5, num_post_neurons, num_pre_neurons)
        applier_args = {
            "x0": x0[np.newaxis, :],
            "y0": y0[:, np.newaxis],
            "u": u,
            "weights": w,
            "traces": traces,
            # Adding numpy to applier args to be able to use it for sign method
            "np": np,
            "x1_y0": traces[1, 0],
            "y1_x0": traces[0, 2],
            "x2_u": x2_u
        }

        result = w + (np.broadcast_to(x0[np.newaxis, :], conn_shape) * (-1)
                      * (2 ** -1)
                      * np.broadcast_to(y1_x0[:, np.newaxis], conn_shape)
                      + np.broadcast_to(y0[:, np.newaxis], conn_shape) * 2
                      * np.broadcast_to(x1_y0[np.newaxis, :], conn_shape) * w
                      + u * np.broadcast_to(x2_u[np.newaxis, :], conn_shape)
                      + u * np.sign(w + 4))

        np.testing.assert_array_equal(
            learning_rule_applier.apply(w, **applier_args), result)


class TestLearningRuleFixedBitApproxApplier(unittest.TestCase):
    def test_learning_rule_bit_approx_applier(self) -> None:
        """Known value test for LearningRuleApplierBitApprox's apply method."""
        default_rng = np.random.default_rng(seed=0)

        target = "dw"
        str_learning_rule = \
            'x0*(-1)*2^-1*y1 + y0*1*2^1*x1*w + u0*x2 + u0*sgn(w+4)'

        u = 1

        x0 = np.array([0, 0, 1, 1, 0], dtype=int)
        x_shape = x0.shape
        x1_y0 = default_rng.integers(low=0, high=17, size=x_shape, dtype=int)
        x2_u = default_rng.integers(low=0, high=17, size=x_shape, dtype=int)

        y0 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=int)
        y_shape = y0.shape
        y1_x0 = default_rng.integers(low=0, high=17, size=y_shape, dtype=int)

        conn_shape = (y0.shape[0], x0.shape[0])
        w = default_rng.integers(low=0, high=51, size=conn_shape, dtype=int)

        traces = np.zeros((3, 5) + conn_shape, dtype=int)
        traces[0, 2, :, :] = np.broadcast_to(y1_x0[:, np.newaxis], conn_shape)
        traces[1, 0, :, :] = np.broadcast_to(x1_y0[np.newaxis, :], conn_shape)
        traces[2, 1, :, :] = np.broadcast_to(x2_u[np.newaxis, :], conn_shape)

        se = SymbolicEquation(target, str_learning_rule)
        product_series = ProductSeries(se)

        learning_rule_applier = LearningRuleApplierBitApprox(product_series)

        self.assertIsInstance(learning_rule_applier,
                              LearningRuleApplierBitApprox)

        # Shape x0: (num_pre_neurons) -> (1, num_pre_neurons)
        # Shape y0: (num_post_neurons) -> (num_post_neurons, 1)
        # Shape w: (num_post_neurons, num_pre_neurons)
        # Shape traces : (3, 5, num_post_neurons, num_pre_neurons)
        applier_args = {
            "shape": conn_shape,
            "x0": x0[np.newaxis, :],
            "y0": y0[:, np.newaxis],
            "u": u,
            "weights": w,
            "x1_y0": traces[1, 0],
            "y1_x0": traces[0, 2],
            "x2_u": x2_u
        }

        result = [[5295, 1294, 2025, 738, 1792], [2196, 3627, 988, -575, 3367],
                  [165, 299, -376, -124, 1836], [129, 2331, -252, 15, 2584],
                  [149, 276, -895, -640, 1798], [128, 2850, 922, 673, 2317],
                  [159, 294, -493, -233, 1842], [169, 306, -429, -158, 1840],
                  [3745, 3626, 1635, 867, 2323]]

        np.testing.assert_array_equal(
            learning_rule_applier.apply(w, **applier_args), result)


if __name__ == "__main__":
    unittest.main()
