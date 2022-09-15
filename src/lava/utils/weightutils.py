# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from enum import Enum, unique

import numpy as np


@unique
class SignMode(Enum):
    """Enumeration of sign mode of weights."""

    NULL = 0
    MIXED = 1
    EXCITATORY = 2
    INHIBITORY = 3


def optimize_weight_bits(
    weight: np.ndarray, loihi2: bool = False
) -> ty.Tuple[np.ndarray, int, int, SignMode]:
    """Optimizes the weight matrix to best fit in Loihi's synapse.

    Parameters
    ----------
    weight : np.ndarray
        Standard 8 bit signed weight matrix.
    loihi2 : bool, optional
        Flag to optimize for Lohi 2. By default False.

    Returns
    -------
    np.ndarray
        optimized weight matrix
    int
        weight bits
    int
        weight_exponent
    SignMode
        synapse sign mode
    """
    max_weight = np.max(weight)
    min_weight = np.min(weight)

    if max_weight < 0:
        sign_mode = SignMode.INHIBITORY
        is_signed = 0
    elif min_weight >= 0:
        sign_mode = SignMode.EXCITATORY
        is_signed = 0
    else:
        sign_mode = SignMode.MIXED
        is_signed = 1

    if sign_mode == SignMode.MIXED:
        pos_scale = 127 / max_weight
        neg_scale = -128 / min_weight
        scale = np.min([pos_scale, neg_scale])
    elif sign_mode == SignMode.INHIBITORY:
        scale = -256 / min_weight
    elif sign_mode == SignMode.EXCITATORY:
        scale = 255 / max_weight

    scale_bits = int(np.floor(np.log2(scale)) + is_signed)

    precision_found = False
    n = 8
    while (precision_found is False) and (n > 0):
        roundingError = np.sum(
            np.abs(weight / (2**n) - np.round(weight / (2**n)))
        )
        if roundingError == 0:
            precision_found = True
        else:
            n -= 1

    n -= is_signed

    num_weight_bits = 8 - scale_bits - n
    weight_exponent = -scale_bits

    weight = np.left_shift(weight.astype(np.int32), int(scale_bits))

    if loihi2:
        weight = weight // (1 << (8 - num_weight_bits))
        if sign_mode == SignMode.MIXED:
            weight = weight // 2

    return (
        weight.astype(int),
        int(num_weight_bits),
        int(weight_exponent),
        sign_mode,
    )
