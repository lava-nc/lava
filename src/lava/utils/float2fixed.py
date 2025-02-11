from typing import *
import numpy as np
from numba import njit


@njit
def get_integer_precision(val: np.ndarray) -> int:
    """Returns precision of integer input `val`."""
    return 8 * val.itemsize


@njit
def get_msb_pos(val: np.ndarray) -> np.ndarray:
    """Calculates the MSB of `val`, using pure bit-shifts. 
    This is equivalent to floor(log2(val)). Consequently, 
    it assumes val > 0.
    """
    result = np.zeros_like(val)
    for i in range(val.size):
        v = val.flat[i]
        if v == 0:
            result.flat[i] = 0
        else:
            ret_val = 0
            while v > 0:
                v = v >> 1
                ret_val += 1
            result.flat[i] = ret_val
    return result


@njit
def msb_align_to_s32bit(val: np.ndarray) -> np.ndarray:
    """Self explanatory"""
    sgn = np.where(val < 0, -1, 1)
    abs_val = np.abs(val)
    msb = get_msb_pos(abs_val)
    return sgn * left_shift(abs_val, 31 - msb)


@njit
def left_shift(val: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Shift function allowing negative shifts. If `shift` is negative,
    it is interpreted as right-shift. CAUTION: Overflows beyond int32
    are not guarded. Values will wrap around min and max of int32.
    """
    shift_neg = np.where(shift < 0, np.abs(shift), 0)
    val_neg = np.where(shift < 0, val, 0)
    shift_pos = np.where(shift > 0, shift, 0)
    val_pos = np.where(shift >= 0, val, 0)

    ret_val_neg = val_neg >> shift_neg
    ret_val_pos = val_pos << shift_pos

    return ret_val_neg + ret_val_pos


@njit
def right_shift(val: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Shift function allowing negative shifts. If `shift` is negative,
    it is interpreted as left-shift. CAUTION: Overflows beyond int32
    are not guarded. Values will wrap around min and max of int32.
    """
    shift_neg = np.where(shift < 0, np.abs(shift), 0)
    val_neg = np.where(shift < 0, val, 0)
    shift_pos = np.where(shift > 0, shift, 0)
    val_pos = np.where(shift >= 0, val, 0)

    ret_val_neg = val_neg << shift_neg
    ret_val_pos = val_pos >> shift_pos

    return ret_val_neg + ret_val_pos


@njit
def calc_fraction_and_mantissa(val: np.ndarray, 
                               precision: int) -> Tuple[np.ndarray, 
                                                        np.ndarray]:
    """Calculates fractional part and mantissa by right-shifting `val` by
    `precision`. Assumes that `val` is positive.
    """
    fraction = val & (2 ** precision - 1)
    mantissa = val >> precision

    return fraction, mantissa


@njit
def calc_signed_range(precision: int) -> Tuple[int, int]:
    """Calculates min and max range for the specified precision.
    """
    two_raised_to_precision_m_1 = 1 << (precision - 1)
    min_of_range = -two_raised_to_precision_m_1
    max_of_range = two_raised_to_precision_m_1 - 1

    return min_of_range, max_of_range


@njit
def calc_unsigned_range(precision: int) -> Tuple[int, int]:
    """Calculates min and max range for the specified `precision`.
    """
    min_of_range = 0
    max_of_range = (1 << precision) - 1

    return min_of_range, max_of_range


@njit
def clip_to_arbit_prec(val: np.ndarray, target_precision: int) -> np.ndarray:
    """Clips input `val` to range dictated by the specified `target_precision`.
    """
    range_min, range_max = calc_signed_range(target_precision)
    return np.clip(val, range_min, range_max)


@njit
def cast_to_arbit_prec_signed(val: np.ndarray, 
                              target_precision: int) -> np.ndarray[int]:
    """Casts input `val` to arbitrary `target_precision`. Overflows 
    are handled by periodic boundary condition, i.e., large positive 
    numbers 'wrap-around' and become negative and vice versa.
    """
    range_min, range_max = calc_signed_range(target_precision)
    precision_range = range_max - range_min + 1

    pos_val = np.where(val > 0, val, 0)
    neg_val = np.where(val < 0, -val, 0)

    pos_val_mod_range = pos_val % precision_range
    neg_val_mod_range = neg_val % precision_range

    pos_val_cast = np.where(pos_val_mod_range <= range_max, 
                            pos_val_mod_range, 
                            pos_val_mod_range - precision_range)
    neg_val_cast = np.where(-neg_val_mod_range >= range_min, 
                            -neg_val_mod_range, 
                            precision_range - neg_val_mod_range)
    
    ret_val = pos_val_cast.astype(np.int_) + neg_val_cast.astype(np.int_)

    return ret_val


@njit
def stochastic_round(val: np.ndarray, w_frac: int) -> np.ndarray:
    """Stochastically rounds-down `val` by `w_frac` bits.
    """
    sgn_mask = np.where(val < 0, -1, 1)
    abs_val = np.abs(val)

    fraction, mantissa = calc_fraction_and_mantissa(abs_val, w_frac)

    val_rand = np.random.randint(0, 2 ** w_frac, size=val.shape)
    val_incr = np.where(val_rand < fraction, 1, 0)
    val_round = mantissa + val_incr

    return sgn_mask * val_round


@njit
def stochastic_round_wgts(val: np.ndarray, 
                          w_frac: int, 
                          sign_mode: np.ndarray, 
                          num_wgt_bits: np.ndarray) -> np.ndarray:
    """Stochastically rounds-down `val` by `w_frac` bits.
    """
    sgn_mask = np.where(val < 0, -1, 1)
    abs_val = np.abs(val)

    fraction, mantissa = calc_fraction_and_mantissa(abs_val, w_frac)

    mixed_idx = np.where(sign_mode == 1, 1, 0)
    incr = 1 << (mixed_idx + 8 - num_wgt_bits)

    val_rand = np.random.randint(0, 2 ** w_frac, size=val.shape)
    val_incr = np.where(val_rand < fraction, incr, 0)
    val_round = mantissa + val_incr

    return sgn_mask * val_round


# @njit
def stochastic_round_to_target_precision(val: np.ndarray, 
                                         original_precision: int, 
                                         target_precision: int
                                         ) -> Tuple[np.ndarray, int]:
    """Stochastically round `val` to `target_precision` and also return a
    scaling exponent delta_precision, where
    delta_precision = original_precision - target_precision.
    """
    r1 = val > (2 ** (original_precision - 1) - 1)
    r2 = val < - (2 ** (original_precision - 1))
    if np.any(np.logical_or(r1, r2)):
        raise OverflowError(
            "Values overflowing signed original precision found")
    s1 = val > (2 ** (target_precision - 1) - 1)
    s2 = val < - (2 ** (target_precision - 1))
    need_to_round_mask = np.logical_or(s1, s2)
    delta_precision = \
        original_precision - target_precision + 1  # +1 for signed
    ret_val = stochastic_round(val * need_to_round_mask, delta_precision)

    ret_val = ret_val + (1 - need_to_round_mask) * val

    return ret_val, delta_precision


@njit
def nearest_round(val: np.ndarray, w_frac: int) -> np.ndarray:
    """Rounds `val` to the nearest `target_precision` integer and 
    also returns a scaling exponent delta_precision, where
    delta_precision = original_precision - target_precision.
    """
    sgn_mask = np.where(val < 0, -1, 1)
    abs_val = np.abs(val)

    fraction = abs_val & (2 ** w_frac - 1)
    mantissa = abs_val >> w_frac

    val_flag = fraction >> (w_frac - 1)
    val_incr = np.where(val_flag == 1, 1, 0)
    val_round = mantissa + val_incr

    return sgn_mask * val_round


@njit
def nearest_round_to_target_precision(val: np.ndarray, 
                                      original_precision: int, 
                                      target_precision: int
                                      ) -> Tuple[np.ndarray, int]:
    """Rounds `val` to the nearest `target_precision` integer and 
    also returns a scaling exponent delta_precision, where
    delta_precision = original_precision - target_precision.
    """
    r1 = val > (2 ** (original_precision - 1) - 1)
    r2 = val < - (2 ** (original_precision - 1))
    if np.any(np.logical_or(r1, r2)):
        raise OverflowError(
            "Values overflowing signed original precision found")
    s1 = val > (2 ** (target_precision - 1) - 1)
    s2 = val < - (2 ** (target_precision - 1))
    need_to_round_mask = np.logical_or(s1, s2)
    delta_precision = \
        original_precision - target_precision + 1  # +1 for signed
    ret_val = nearest_round(val * need_to_round_mask, delta_precision)

    ret_val = ret_val + (1 - need_to_round_mask) * val

    return ret_val, delta_precision


@njit
def float_to_fixed_stochastic_round(val: np.ndarray, 
                                    min_dyn_range: float, 
                                    max_dyn_range: float, 
                                    target_precision: int) -> np.ndarray:
    """Casts a float32 input `val` of dynamical range between `min_dyn_range`
    and `max_dyn_range` to fixed point representation with `target_precision`.
    """
    if min_dyn_range < 0:
        if max_dyn_range > 0:
            min_fp_range, max_fp_range = calc_signed_range(target_precision)
        else:
            min_fp_range, max_fp_range = calc_unsigned_range(target_precision)
            min_fp_range -= 1
            max_fp_range = -max_fp_range
    else:
        min_fp_range, max_fp_range = calc_unsigned_range(target_precision)
        if min_dyn_range > 0:
            min_fp_range += 1

    fp_range = max_fp_range - min_fp_range
    dyn_range = max_dyn_range - min_dyn_range

    intermediate_val = (fp_range / dyn_range) * (val - min_dyn_range) + \
        min_fp_range

    sgn_mask = np.where(intermediate_val < 0, -1, 1)
    abs_val = np.abs(intermediate_val)

    int_part = np.floor(abs_val)
    frc_part = sgn_mask * intermediate_val - int_part

    val_rand = np.random.uniform(0, 1, size=val.shape)
    val_incr = np.where(val_rand < frc_part, 1, 0)
    val_round = int_part + val_incr

    return sgn_mask * val_round


@njit
def float_to_fixed_nearest_round(val: np.ndarray, 
                                 min_dyn_range: float, 
                                 max_dyn_range: float, 
                                 target_precision: int) -> np.ndarray:
    """Casts a float32 input `val` of dynamical range between `min_dyn_range`
    and `max_dyn_range` to fixed point representation with `target_precision`
    """
    if min_dyn_range < 0:
        if max_dyn_range > 0:
            min_fp_range, max_fp_range = calc_signed_range(target_precision)
        else:
            min_fp_range, max_fp_range = calc_unsigned_range(target_precision)
            min_fp_range -= 1
            max_fp_range = -max_fp_range
    else:
        min_fp_range, max_fp_range = calc_unsigned_range(target_precision)
        if min_dyn_range > 0:
            min_fp_range += 1

    fp_range = max_fp_range - min_fp_range
    dyn_range = max_dyn_range - min_dyn_range

    intermediate_val = (fp_range / dyn_range) * (val - min_dyn_range) + \
        min_fp_range

    sgn_mask = np.where(intermediate_val < 0, -1, 1)
    abs_val = np.abs(intermediate_val)

    int_part = np.floor(abs_val)
    frc_part = sgn_mask * intermediate_val - int_part

    val_incr = np.where(0.5 < frc_part, 1, 0)
    val_round = int_part + val_incr

    return sgn_mask * val_round


@njit
def split_to_mantissa_exponent_master(val: np.ndarray, 
                                      max_val: np.ndarray, 
                                      w_mantissa: int, 
                                      w_exponent: int, 
                                      split_by_value: bool, 
                                      range_is_static: bool
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Splits input `val` into a mantissa and an exponent tensor, such that
    `val` is approximately mantissa * (2 ** exponent)
    """
    # log_of_2 = np.log(2.0)
    val_f = val.astype(np.float32)
    if split_by_value:
        max_val[:] = np.abs(val_f)
    else:
        if range_is_static:
            max_val[:] = np.max(val_f)
            
    val_for_expt_computation = max_val

    log_2_val_for_expt = np.log2(val_for_expt_computation)  # / log_of_2

    val_expt = np.ceil(log_2_val_for_expt) - w_mantissa + 1
    val_mant = np.floor(val_f / (2.0 ** val_expt))

    val_mant = np.where(val == 0, 0, val_mant)
    val_expt = np.where(val == 0, 0, val_expt)

    val_mant = clip_to_arbit_prec(val_mant, w_mantissa)
    val_expt = clip_to_arbit_prec(val_expt, w_exponent)

    return val_mant, val_expt


@njit
def split_to_mantissa_exponent_by_value(val: np.ndarray, 
                                        w_mantissa: int, 
                                        w_exponent: int
                                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Splits input `val` into a mantissa and an exponent tensor, such that
    `val` is approximately mantissa * (2 ** exponent)
    """
    return split_to_mantissa_exponent_master(val, 
                                             val, 
                                             w_mantissa, 
                                             w_exponent, 
                                             True, 
                                             False)


@njit
def split_to_mantissa_exponent_dynamic_range(val: np.ndarray, 
                                             max_val: np.ndarray, 
                                             w_mantissa: int, 
                                             w_exponent: int
                                             ) -> Tuple[np.ndarray, 
                                                        np.ndarray]:
    """Splits input `val` into a mantissa and an exponent tensor, such that
    `val` is approximately mantissa * (2 ** exponent). Takes into account 
    the dynamic range of `val` by explicitly using `max_val`,
    the maximum value any component of `val` would take in its lifetime
    """
    return split_to_mantissa_exponent_master(val, 
                                             max_val, 
                                             w_mantissa, 
                                             w_exponent, 
                                             False, 
                                             False)


@njit
def split_to_mantissa_exponent_static_range(val: np.ndarray, 
                                            w_mantissa: int, 
                                            w_exponent: int
                                            ) -> Tuple[np.ndarray, 
                                                       np.ndarray]:
    """Splits input `val` into a mantissa and an exponent tensor, such that
    `val` is approximately mantissa * (2 ** exponent). Takes into account 
    the static range of `val` by implicitly using max(`val`),
    the maximum of the components of `val`
    """
    return split_to_mantissa_exponent_master(val, 
                                             val, 
                                             w_mantissa, 
                                             w_exponent, 
                                             False, 
                                             True)