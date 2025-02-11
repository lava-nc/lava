import unittest
from test import support

import numpy as np
from numba import njit

import lava.utils.float2fixed as fp


class TestFixedPoint(unittest.TestCase):
    def test_get_integer_precision(self):
        """Checks retrieval of integer precision"""
        num8 = np.array([7], dtype=np.int8)
        num16 = np.array([7], dtype=np.int16)
        num32 = np.array([7], dtype=np.int32)
        num64 = np.array([7], dtype=np.int64)

        unum8 = np.array([7], dtype=np.uint8)
        unum16 = np.array([7], dtype=np.uint16)
        unum32 = np.array([7], dtype=np.uint32)
        unum64 = np.array([7], dtype=np.uint64)

        precisions = []
        for num in [num8, num16, num32, num64, unum8, unum16, unum32, unum64]:
            precisions.append(fp.get_integer_precision(num))

        gold_std = np.array([8, 16, 32, 64, 8, 16, 32, 64], 
                            dtype=np.int32)
        np.testing.assert_array_equal(gold_std, np.array(precisions, 
                                                         dtype=np.int32))

    def test_custom_shift(self):
        """Checks correctness of generalised left-shift"""

        shift_l = np.array([[[ 0,  1,  0, -2,  1],
                             [-2,  0,  0,  0,  0],
                             [-1, -1,  2,  0,  2],
                             [-1,  0,  1, -1, -1]],
                            [[ 2,  2, -1,  0,  0],
                             [ 1, -2,  2,  2,  1],
                             [-2, -1,  0,  0,  1],
                             [ 1, -2,  0, -1, -1]],
                            [[ 0, -1,  1, -1,  1],
                             [-2, -1, -2,  2,  0],
                             [ 1, -1,  1, -2,  2],
                             [ 0,  0,  0,  2,  1]]], dtype=np.int32)
        shift_r = -shift_l
        val = np.full((3, 4, 5), 16, dtype=np.int32)
        ret_val_l = fp.left_shift(val, shift_l)
        ret_val_r = fp.right_shift(val, shift_r)
        gold_std = np.array([[[16, 32, 16,  4, 32],
                              [ 4, 16, 16, 16, 16],
                              [ 8,  8, 64, 16, 64],
                              [ 8, 16, 32,  8,  8]],
                             [[64, 64,  8, 16, 16],
                              [32,  4, 64, 64, 32],
                              [ 4,  8, 16, 16, 32],
                              [32,  4, 16,  8,  8]],
                             [[16,  8, 32,  8, 32],
                              [ 4,  8,  4, 64, 16],
                              [32,  8, 32,  4, 64],
                              [16, 16, 16, 64, 32]]], dtype=np.int32)
        np.testing.assert_array_equal(gold_std, ret_val_l)
        np.testing.assert_array_equal(gold_std, ret_val_r)

    def test_calc_signed_range(self):
        """Checks calculation of the range for signed integers of a given
        precision"""
        precision = 24
        gold_std = (-2 ** 23, 2 ** 23 - 1)
        min_range, max_range = fp.calc_signed_range(precision)

        self.assertEqual(gold_std, (min_range, max_range))

    @unittest.skip
    def test_calc_unsigned_range(self):
        """Checked calculation of the range for unsigned integers of a given
        precision"""
        precision = 24
        min_gold_std = 0
        max_gold_std = 2 ** 24 - 1
        min_range, max_range = fp.calc_unsigned_range(precision)

        min_good = not (min_gold_std > min_range or min_gold_std < min_range)
        max_good = not (max_gold_std > max_range or max_gold_std < max_range)

        self.assertTrue(min_good and max_good)

    def test_cast_int_1(self):
        """Check -ve integer casting from 32-bit to 24-bit
        (both represented as 32-bit)"""
        val = np.array([-4353], dtype=np.int32)
        precision = 24

        gold_std = val
        cast_val = fp.clip_to_arbit_prec(val, target_precision=precision)

        np.testing.assert_array_equal(gold_std, cast_val)

    def test_cast_int_2(self):
        """Check -ve integer casting from 32-bit to 24-bit
        (the latter represented as 32-bit)"""
        val = np.array([-(2 ** 27 + 2 ** 13 + 1)], dtype=np.int32)
        precision = 24

        # Expect the cast_val to saturate to the -ve limit
        gold_std = np.array([-2 ** 23], dtype=np.int32)
        cast_val = fp.clip_to_arbit_prec(val, target_precision=precision)

        np.testing.assert_array_equal(gold_std, cast_val)

    def test_cast_int_3(self):
        """Check +ve integer casting from 32-bit to 24-bit
         the latter represented as 32-bit)"""
        val = np.array([2 ** 25 + 2 ** 14 + 2 ** 7], dtype=np.int32)
        precision = 24

        gold_std = np.array([2 ** 23 - 1], dtype=np.int32)
        cast_val = fp.clip_to_arbit_prec(val, target_precision=precision)

        np.testing.assert_array_equal(gold_std, cast_val)

    @unittest.skip
    def test_cast_int_4(self):
        """Check unsigned integer casting from 32-bit to 24-bit
        (the latter represented as 32-bit)"""
        # val is greater than max of int32 but much smaller than max of uint32
        val = np.array([2 ** 23 + 2 ** 4 + 1], dtype=np.uint32)
        precision = 24

        gold_std = val
        cast_val = fp.clip_to_arbit_prec(val, target_precision=precision)

        self.assertTrue(np.all(gold_std == cast_val))

    def test_cast_to_arbit_prec(self):
        """Checks casting to an arbitrary precision"""
        val = np.array([-932.3324], dtype=np.float32)
        prec = 10

        ret_val = fp.cast_to_arbit_prec_signed(val=val, target_precision=prec)

        self.assertEqual(ret_val, 91)

    def test_stochastic_round(self):
        """Check stochastic rounding functionality"""
        val = np.array([715130017], dtype=np.int32)
        orig_prec = 32
        precision = 24

        out_val, exp = fp.stochastic_round_to_target_precision(val, 
                                                               orig_prec, 
                                                               precision)

        self.assertTrue(bin(val[0] >> 9) == bin(out_val[0] >> 1) and exp == 8)

    def test_nearest_round(self):
        """Check rounding to the nearest integer"""
        val = np.array([715130017], dtype=np.int32)
        orig_prec = 32
        precision = 24

        out_val, exp = fp.nearest_round_to_target_precision(val, 
                                                            orig_prec, 
                                                            precision)

        self.assertTrue(bin(val[0] >> 9) == bin(out_val[0] >> 1) and exp == 8)

    def test_float_to_fixed_stochastic_round(self):
        """Check stochastic rounding of a float32 to
        an integer of desired precision"""

        np.random.seed(0)

        val = np.array([-0.15], dtype=np.float32)
        min_dyn_range = np.array([-0.2], dtype=np.float32)
        max_dyn_range = np.array([0.2], dtype=np.float32)
        precision = 8

        gold_std = np.array([-96], dtype=np.int32)
        ret_val = fp.float_to_fixed_stochastic_round(val, 
                                                     min_dyn_range, 
                                                     max_dyn_range, 
                                                     precision)

        np.testing.assert_array_equal(gold_std, ret_val)

    def test_split_to_mantissa_exponent_by_value(self):
        """Checks splitting of a float into mantissa and exponent"""

        val = np.array([0.00167484])
        w_mant = 2
        w_expt = 5
        mant, expt = fp.split_to_mantissa_exponent_by_value(val, 
                                                            w_mantissa=w_mant, 
                                                            w_exponent=w_expt)

        self.assertEqual(expt[0], -10)
        self.assertEqual(mant[0], 1)

    def test_split_to_mantissa_exponent_dynamic_range(self):
        """Checks splitting of a float into mantissa and exponent"""

        val = np.array([0.00167484])
        w_mant = 3
        w_expt = 5
        mant, expt = \
            fp.split_to_mantissa_exponent_dynamic_range(
                val, 
                max_val=np.array([0.002]),
                w_mantissa=w_mant, 
                w_exponent=w_expt)

        self.assertEqual(expt[0], -10)
        self.assertEqual(mant[0], 1)

    def test_split_to_mantissa_exponent_static_range(self):
        """Checks splitting of a float into mantissa and exponent"""

        val = np.array([0.00167484, 0.002, 0.000983, -0.001832], 
                       dtype=np.float32)
        w_mant = 3
        w_expt = 5
        mant, expt = \
            fp.split_to_mantissa_exponent_static_range(
                val, 
                w_mantissa=w_mant, 
                w_exponent=w_expt)

        np.testing.assert_array_equal(expt, np.array([-10, -10, -10, -10]))
        np.testing.assert_array_equal(mant, np.array([1, 2, 1, -2]))
