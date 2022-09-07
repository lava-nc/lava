# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np


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
