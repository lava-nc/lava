# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


def saturate(min_value: int, values: np.ndarray, max_value: int) -> np.ndarray:
    """Saturate ndarray given minimum and maximum values.

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


def stochastic_round(values: np.ndarray,
                     random_numbers: ty.Union[int, float, np.ndarray],
                     probabilities: np.ndarray) -> np.ndarray:
    """Stochastically add 1 to an ndarray at location where random numbers are
    less than given probabilities.

    Parameters
    ----------
    values: ndarray
        Values before stochastic rounding.
    random_numbers: int or float or ndarray
        Randomly generated number or ndarray of numbers.
    probabilities: ndarray
        Probabilities to stochastically round.

    Returns
    ----------
    result : ndarray
        Stochastically rounded values.
    """
    return (values + (random_numbers < probabilities).astype(int)).astype(int)


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
