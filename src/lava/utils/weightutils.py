# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from enum import Enum, unique
from dataclasses import dataclass


@unique
class SignMode(Enum):
    """Enumeration of sign mode of weights."""
    NULL = 0
    MIXED = 1
    EXCITATORY = 2
    INHIBITORY = 3


def determine_sign_mode(weights: np.ndarray) -> SignMode:
    """Determines the sign mode that describes the values in the given
    weight matrix.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix

    Returns
    -------
    SignMode
        The sign mode that best describes the values in the given weight
        matrix.
    """
    if np.min(weights) >= 0:
        sign_mode = SignMode.EXCITATORY
    elif np.max(weights) <= 0:
        sign_mode = SignMode.INHIBITORY
    else:
        sign_mode = SignMode.MIXED

    return sign_mode


@dataclass
class OptimizedWeights:
    weights: np.ndarray
    num_weight_bits: int
    weight_exp: int


def optimize_weight_bits(
        weights: np.ndarray,
        sign_mode: SignMode,
        loihi2: ty.Optional[bool] = False) -> OptimizedWeights:
    """Optimizes the weight matrix to best fit in Loihi's synapse.

    Parameters
    ----------
    weights : np.ndarray
        Standard 8-bit signed weight matrix.
    sign_mode : SignMode
        Determines whether the weights are purely excitatory, inhibitory,
        or mixed sign.
    loihi2 : bool, optional
        Flag to optimize for Loihi 2. By default False.

    Returns
    -------
    OptimizedWeights
        An object that wraps the optimized weight matrix and weight parameters.
    """
    _validate_weights(weights, sign_mode)

    weight_exp = _determine_weight_exp(weights, sign_mode)
    num_weight_bits = _determine_num_weight_bits(weights, weight_exp, sign_mode)

    weights = np.left_shift(weights.astype(np.int32), int(-weight_exp))

    if loihi2:
        weights = weights // (1 << (8 - num_weight_bits))
        if sign_mode == SignMode.MIXED:
            weights = weights // 2

    optimized_weights = OptimizedWeights(weights=weights.astype(int),
                                         num_weight_bits=num_weight_bits,
                                         weight_exp=weight_exp)
    return optimized_weights


def _validate_weights(weights: np.ndarray,
                      sign_mode: SignMode) -> None:
    """Validate the weight values against the given sign mode.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix
    sign_mode : SignMode
        Sign mode specified for the weight matrix
    """
    mixed_flag = int(sign_mode == SignMode.MIXED)
    excitatory_flag = int(sign_mode == SignMode.EXCITATORY)
    inhibitory_flag = int(sign_mode == SignMode.INHIBITORY)

    min_weight = (-2 ** 8) * (mixed_flag + inhibitory_flag)
    min_weight += inhibitory_flag
    max_weight = (2 ** 8 - 1) * (mixed_flag + excitatory_flag)

    if np.any(weights > max_weight) or np.any(weights < min_weight):
        raise ValueError(f"weights have to be between {min_weight} and "
                         f"{max_weight} for {sign_mode=}. Got "
                         f"weights between {np.min(weights)} and "
                         f"{np.max(weights)}.")


def _determine_weight_exp(weights: np.ndarray,
                          sign_mode: SignMode) -> int:
    """Determines the weight exponent to be used to optimally represent the
    given weight values and sign mode on Loihi.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix
    sign_mode : SignMode
        The sign mode describing the range of values in the weight matrix.

    Returns
    -------
    int
        Optimal weight exponent for representing the weights on Loihi.
    """
    max_weight = np.max(weights)
    min_weight = np.min(weights)

    scale = 0

    if max_weight == min_weight == 0:
        weight_exp = -0
        return weight_exp

    if sign_mode == SignMode.MIXED:
        pos_scale = 127 / max_weight if max_weight > 0 else np.inf
        neg_scale = -128 / min_weight if min_weight < 0 else np.inf
        scale = np.min([pos_scale, neg_scale])
    elif sign_mode == SignMode.INHIBITORY:
        scale = -255 / min_weight
    elif sign_mode == SignMode.EXCITATORY:
        scale = 255 / max_weight

    scale_bits = int(np.floor(np.log2(scale)))
    if sign_mode == SignMode.MIXED:
        scale_bits += 1

    weight_exp = -scale_bits

    return weight_exp


def _determine_num_weight_bits(weights: np.ndarray,
                               weight_exp: int,
                               sign_mode: SignMode) -> int:
    """Determines the number of bits required to optimally represent the
    given weight matrix on Loihi.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix
    weight_exp : int
        Weight exponent
    sign_mode : SignMode
        Sign mode that describes the values in the weight matrix.

    Returns
    -------
    int
        Optimal number of bits to represent the weight matrix on Loihi.
    """
    precision_found = False
    n = 8
    while (precision_found is False) and (n > 0):
        rounding_error = np.sum(
            np.abs(weights / (2 ** n) - np.round(weights / (2 ** n)))
        )
        if rounding_error == 0:
            precision_found = True
        else:
            n -= 1

    if sign_mode == SignMode.MIXED:
        n -= 1

    num_weight_bits = 8 + weight_exp - n

    return num_weight_bits


def truncate_weights(weights: np.ndarray,
                     sign_mode: SignMode,
                     num_weight_bits: int,
                     max_num_weight_bits: ty.Optional[int] = 8) -> np.ndarray:
    """Truncate the least significant bits of the weight matrix given the
    sign mode and number of weight bits.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix that is to be truncated.
    sign_mode : SignMode
        Sign mode to use for truncation. See SignMode class for the
        correct values.
    num_weight_bits : int
        Number of bits to use for the weight matrix.
    max_num_weight_bits : int, optional
        Maximum number of bits that can be used to represent weights. Default
        is 8.

    Returns
    -------
    numpy.ndarray
        Truncated weight matrix.
    """
    weights = np.copy(weights).astype(np.int32)

    if sign_mode == SignMode.INHIBITORY:
        weights = -weights

    mixed_flag = int(sign_mode == SignMode.MIXED)
    num_truncate_bits = max_num_weight_bits - num_weight_bits + mixed_flag

    truncated_weights = np.left_shift(
        np.right_shift(weights, num_truncate_bits),
        num_truncate_bits).astype(np.int32)

    if sign_mode == SignMode.INHIBITORY:
        truncated_weights = -truncated_weights

    return truncated_weights


def clip_weights(weights: np.ndarray,
                 sign_mode: SignMode,
                 num_bits: int) -> np.ndarray:
    """Truncate the least significant bits of the weight matrix given the
    sign mode and number of weight bits.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix that is to be truncated.
    sign_mode : SignMode
        Sign mode to use for truncation.
    num_bits : int
        Number of bits to use to clip the weights to.

    Returns
    -------
    numpy.ndarray
        Truncated weight matrix.
    """
    weights = np.copy(weights).astype(np.int32)

    mixed_flag = int(sign_mode == SignMode.MIXED)
    inhibitory_flag = int(sign_mode == SignMode.INHIBITORY)

    if inhibitory_flag:
        weights = -weights

    min_wgt = (-2 ** num_bits) * mixed_flag
    max_wgt = 2 ** num_bits - 1

    clipped_weights = np.clip(weights, min_wgt, max_wgt)

    if inhibitory_flag:
        clipped_weights = -clipped_weights

    return clipped_weights
