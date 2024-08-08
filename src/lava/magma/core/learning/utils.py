# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


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


def apply_mask(item: ty.Union[np.ndarray, int], nb_bits: int) \
        -> ty.Union[np.ndarray, int]:
    """Get nb_bits least-significant bits.

    Parameters
    ----------
    item : np.ndarray or int
        Item to apply mask to.
    nb_bits : int
        Number of LSBs to keep.

    Returns
    ----------
    result : np.ndarray or int
        Least-significant bits.
    """
    mask = ~(~0 << nb_bits)
    return item & mask


def float_to_literal(learning_parameter: float) -> str:
    """Convert the floating point representation of the
    learning parameter to the form mantissa * 2 ^ [+/1]exponent.
    Parameters
    ----------
    learning_parameters: float
        the float value of learning-related parameter

    Returns
    -------
    result: str
        string representation of learning_parameter.
    """
    if learning_parameter == 0:
        return "0"

    sign = int(np.sign(learning_parameter))

    learning_parameter = np.abs(learning_parameter)
    mantissa = np.max([int(learning_parameter), 1])
    remainder = learning_parameter / mantissa

    if remainder == 1:
        return f"({sign * mantissa})"

    exp = int(np.round(np.log2(remainder)))
    return f"({sign * mantissa}) * 2 ^ {exp}"
