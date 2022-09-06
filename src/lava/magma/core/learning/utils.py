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
import numpy as np


def get_bit(int_number: int, bit_number: int) -> int:
    """Get nth bit of an integer.

    Parameters
    ----------
    int_number : int
        Integer number.
    bit_number : int
        Bit index.

    Returns
    ----------
    result : int
        Bit value.
    """
    return int_number >> bit_number & 1


def apply_mask(int_number: int, nb_bits: int) -> int:
    """Get nb_bits least-significant bits.

    Parameters
    ----------
    int_number : int
        Integer number.
    nb_bits : int
        Number of LSBs to keep.

    Returns
    ----------
    result : int
        Least-significant bits.
    """

    mask = ~(~0 << nb_bits)

    return int_number & mask


def saturate(min_value: int, values: np.ndarray, max_value: int) -> np.ndarray:
    """Saturate numpy array given minimum and maximum values.

    Parameters
    ----------
    min_value: int
        Minimum value.
    values: ndarray
        Array to saturate.
    max_value : int
        Maximum value.

    Returns
    ----------
    result : ndarray
        Saturated values.
    """
    return np.maximum(min_value, np.minimum(values, max_value))


def stochastic_round(value, random_number, probability):
    return (value + (random_number < probability).astype(int)).astype(int)