# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.utils.weightutils import SignMode, determine_sign_mode,\
    optimize_weight_bits, _determine_weight_exp, \
    _determine_num_weight_bits, truncate_weights, clip_weights


class TestDetermineSignMode(unittest.TestCase):
    def test_determine_sign_mode_inhibitory(self) -> None:
        sign_mode = determine_sign_mode(weights=np.array([-3, -2, -1]))
        self.assertEqual(sign_mode, SignMode.INHIBITORY)

    def test_determine_sign_mode_excitatory(self) -> None:
        sign_mode = determine_sign_mode(weights=np.array([0, 1, 2]))
        self.assertEqual(sign_mode, SignMode.EXCITATORY)

    def test_determine_sign_mode_mixed(self) -> None:
        sign_mode = determine_sign_mode(weights=np.array([-1, 0, 1]))
        self.assertEqual(sign_mode, SignMode.MIXED)


class TestOptimizeWeightBits(unittest.TestCase):
    def test_optimize_raises_error_when_weights_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([256]),
                                 sign_mode=SignMode.EXCITATORY)

        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([-257]),
                                 sign_mode=SignMode.EXCITATORY)

        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([-256]),
                                 sign_mode=SignMode.INHIBITORY)

        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([1]),
                                 sign_mode=SignMode.INHIBITORY)

        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([-257]),
                                 sign_mode=SignMode.MIXED)

        with self.assertRaises(ValueError):
            optimize_weight_bits(weights=np.array([257]),
                                 sign_mode=SignMode.MIXED)

    def test_optimize_weight_bits_excitatory_8bit(self) -> None:
        weights = np.arange(0, 255, 1, dtype=int)
        sign_mode = SignMode.EXCITATORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights, weights)
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 8)

    def test_optimize_weight_bits_excitatory_7bit(self) -> None:
        weights = np.arange(0, 255, 2, dtype=int)
        sign_mode = SignMode.EXCITATORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights,
                                      np.right_shift(weights, 1))
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 7)

    def test_optimize_weight_bits_inhibitory_8bit(self) -> None:
        weights = np.arange(-255, 0, 1, dtype=int)
        sign_mode = SignMode.INHIBITORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights, weights)
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 8)

    def test_optimize_weight_bits_inhibitory_7bit(self) -> None:
        weights = np.arange(-254, 0, 2, dtype=int)
        sign_mode = SignMode.INHIBITORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights,
                                      np.right_shift(weights, 1))
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 7)

    def test_optimize_weight_bits_mixed_8bit(self) -> None:
        weights = np.arange(-256, 255, 2, dtype=int)
        sign_mode = SignMode.MIXED

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights,
                                      np.right_shift(weights, 1))
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 8)

    def test_optimize_weight_bits_mixed_7bit(self) -> None:
        weights = np.arange(-256, 255, 4, dtype=int)
        sign_mode = SignMode.MIXED

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights,
                                      np.right_shift(weights, 2))
        self.assertEqual(optimized.weight_exp, 0)
        self.assertEqual(optimized.num_weight_bits, 7)

    def test_optimize_weight_bits_weight_exp(self) -> None:
        weights = np.array([10, 10, 10])
        sign_mode = SignMode.EXCITATORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights, np.array([5, 5, 5]))
        self.assertEqual(optimized.weight_exp, -4)
        self.assertEqual(optimized.num_weight_bits, 3)

    def test_optimize_weight_bits_wgts_all_zero(self) -> None:
        weights = np.array([0, 0, 0])
        sign_mode = SignMode.EXCITATORY

        optimized = optimize_weight_bits(weights=weights,
                                         sign_mode=sign_mode,
                                         loihi2=True)

        np.testing.assert_array_equal(optimized.weights, np.array([0, 0, 0]))
        self.assertEqual(optimized.weight_exp, -0)
        self.assertEqual(optimized.num_weight_bits, 0)

    def test_determine_weight_exp_inhibitory_0(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([-255, -128, -1]),
                                           sign_mode=SignMode.INHIBITORY)
        self.assertEqual(weight_exp, 0)

    def test_determine_weight_exp_inhibitory_1(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([-510, -256, -1]),
                                           sign_mode=SignMode.INHIBITORY)
        self.assertEqual(weight_exp, 1)

    def test_determine_weight_exp_excitatory_0(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([0, 128, 255]),
                                           sign_mode=SignMode.EXCITATORY)
        self.assertEqual(weight_exp, 0)

    def test_determine_weight_exp_excitatory_1(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([0, 255, 510]),
                                           sign_mode=SignMode.EXCITATORY)
        self.assertEqual(weight_exp, 1)

    def test_determine_weight_exp_mixed_0(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([-256, 0, 254]),
                                           sign_mode=SignMode.MIXED)
        self.assertEqual(weight_exp, 0)

    def test_determine_weight_exp_mixed_1(self) -> None:
        weight_exp = _determine_weight_exp(weights=np.array([-512, 0, 508]),
                                           sign_mode=SignMode.MIXED)
        self.assertEqual(weight_exp, 1)

    def test_determine_num_weight_bits_inhibitory_8(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-256, -1, 1, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.INHIBITORY
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_inhibitory_8_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-512, -1, 2, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.INHIBITORY
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_inhibitory_7(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-256, -1, 2, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.INHIBITORY
        )
        self.assertEqual(num_weight_bits, 7)

    def test_determine_num_weight_bits_inhibitory_7_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-512, -1, 4, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.INHIBITORY
        )
        self.assertEqual(num_weight_bits, 7)

    def test_determine_num_weight_bits_excitatory_8(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(0, 255, 1, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.EXCITATORY
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_excitatory_8_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(0, 511, 2, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.EXCITATORY
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_excitatory_7(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(0, 255, 2, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.EXCITATORY
        )
        self.assertEqual(num_weight_bits, 7)

    def test_determine_num_weight_bits_excitatory_7_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(0, 511, 4, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.EXCITATORY
        )
        self.assertEqual(num_weight_bits, 7)

    def test_determine_num_weight_bits_mixed_8(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-256, 255, 2, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.MIXED
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_mixed_8_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-512, 511, 4, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.MIXED
        )
        self.assertEqual(num_weight_bits, 8)

    def test_determine_num_weight_bits_mixed_7(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-256, 255, 4, dtype=int),
            weight_exp=0,
            sign_mode=SignMode.MIXED
        )
        self.assertEqual(num_weight_bits, 7)

    def test_determine_num_weight_bits_mixed_7_exp_1(self) -> None:
        num_weight_bits = _determine_num_weight_bits(
            weights=np.arange(-512, 511, 8, dtype=int),
            weight_exp=1,
            sign_mode=SignMode.MIXED
        )
        self.assertEqual(num_weight_bits, 7)


class TestTruncateWeights(unittest.TestCase):
    def test_truncate_weights_excitatory_8(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([253, 254, 255]),
            sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([253, 254, 255])
        )

    def test_truncate_weights_excitatory_7(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([252, 253, 253, 254, 255]),
            sign_mode=SignMode.EXCITATORY,
            num_weight_bits=7
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([252, 252, 252, 254, 254])
        )

    def test_truncate_weights_inhibitory_8(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([-256, -255, -254, -1]),
            sign_mode=SignMode.INHIBITORY,
            num_weight_bits=8
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([-256, -255, -254, -1])
        )

    def test_truncate_weights_inhibitory_7(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([-255, -254, -253, -252, -251]),
            sign_mode=SignMode.INHIBITORY,
            num_weight_bits=7
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([-254, -254, -252, -252, -250])
        )

    def test_truncate_weights_mixed_8(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([-256, -255, -254, 253, 254, 255]),
            sign_mode=SignMode.MIXED,
            num_weight_bits=8
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([-256, -256, -254, 252, 254, 254])
        )

    def test_truncate_weights_mixed_7(self) -> None:
        truncated_weights = truncate_weights(
            weights=np.array([-256, -255, -254, 253, 254, 255]),
            sign_mode=SignMode.MIXED,
            num_weight_bits=7
        )
        np.testing.assert_array_equal(
            truncated_weights,
            np.array([-256, -256, -256, 252, 252, 252])
        )


class TestClipWeights(unittest.TestCase):
    def test_clip_weights_excitatory_8(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-1, 0, 255, 256]),
            sign_mode=SignMode.EXCITATORY,
            num_bits=8
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([0, 0, 255, 255])
        )

    def test_clip_weights_excitatory_7(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-1, 0, 127, 128]),
            sign_mode=SignMode.EXCITATORY,
            num_bits=7
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([0, 0, 127, 127])
        )

    def test_clip_weights_inhibitory_8(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-256, -255, -1, 0, 1]),
            sign_mode=SignMode.INHIBITORY,
            num_bits=8
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([-255, -255, -1, 0, 0])
        )

    def test_clip_weights_inhibitory_7(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-128, -127, -1, 0, 1]),
            sign_mode=SignMode.INHIBITORY,
            num_bits=7
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([-127, -127, -1, 0, 0])
        )

    def test_clip_weights_mixed_8(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-257, -256, 255, 256]),
            sign_mode=SignMode.MIXED,
            num_bits=8
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([-256, -256, 255, 255])
        )

    def test_clip_weights_mixed_7(self) -> None:
        clipped_weights = clip_weights(
            weights=np.array([-129, -128, 127, 128]),
            sign_mode=SignMode.MIXED,
            num_bits=7
        )
        np.testing.assert_array_equal(
            clipped_weights,
            np.array([-128, -128, 127, 127])
        )
