# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from enum import Enum, unique
from dataclasses import dataclass


@unique
class SignMode(Enum):
    """Enumeration of sign mode of weights.
    """
    NULL = 0
    MIXED = 1
    EXCITATORY = 2
    INHIBITORY = 3


def determine_sign_mode(weights: np.ndarray) -> SignMode:
    if np.max(weights) < 0:
        sign_mode = SignMode.INHIBITORY
    elif np.min(weights) >= 0:
        sign_mode = SignMode.EXCITATORY
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
    if np.any(weights > 255) or np.any(weights < -256):
        raise ValueError(f"weights have to be between -256 and 255. Got "
                         f"weights between {np.min(weights)} and "
                         f"{np.max(weights)}.")

    # Determine the exponent required to represent the weight matrix with a
    # weight-matissa of up to 8 bits.
    weight_exp = _determine_weight_exp(weights, sign_mode)
    weights = np.left_shift(weights.astype(np.int32), -weight_exp)

    num_weight_bits = _determine_num_weight_bits(weights, weight_exp, sign_mode)

    if loihi2:
        weights = weights // (1 << (8 - num_weight_bits))
        if sign_mode == SignMode.MIXED:
            weights = weights // 2

    optimized_weights = OptimizedWeights(weights=weights.astype(int),
                                         num_weight_bits=num_weight_bits,
                                         weight_exp=weight_exp)
    return optimized_weights


def _determine_weight_exp(weights: np.ndarray,
                          sign_mode: SignMode) -> int:
    max_weight = np.max(weights)
    min_weight = np.min(weights)

    scale = 0

    if sign_mode == SignMode.MIXED:
        pos_scale = 127 / max_weight if max_weight > 0 else np.inf
        neg_scale = -128 / min_weight if min_weight < 0 else np.inf
        scale = np.min([pos_scale, neg_scale])
    elif sign_mode == SignMode.INHIBITORY:
        scale = -256 / min_weight
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
                     num_weight_bits: int) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray
        Truncated weight matrix.

    """
    weights = np.copy(weights).astype(np.int32)

    mixed_flag = 1 if sign_mode == SignMode.MIXED else 0
    num_truncate_bits = 8 - num_weight_bits + mixed_flag

    # Saturate the weights according to the sign_mode.
    mixed_flag = int(sign_mode == SignMode.MIXED)
    excitatory_flag = int(sign_mode == SignMode.EXCITATORY)
    inhibitory_flag = int(sign_mode == SignMode.INHIBITORY)

    min_wgt = (-2 ** 8) * (mixed_flag + inhibitory_flag)
    max_wgt = (2 ** 8 - 1) * (mixed_flag + excitatory_flag)

    clipped_weights = np.clip(weights, min_wgt, max_wgt)

    truncated_weights = np.left_shift(
        np.right_shift(clipped_weights, num_truncate_bits),
        num_truncate_bits).astype(np.int32)

    return truncated_weights
