# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import numpy as np
from typing import Tuple


def get_coefficients() -> [np.ndarray, np.ndarray, np.ndarray]:
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    "Initialize A, B and C with values trained on efficientnet features."
    s4d_A = np.load(curr_dir + "/s4d_A.dat.npy").flatten()
    s4d_B = np.load(curr_dir + "/s4d_B.dat.npy").flatten()
    s4d_C = np.load(curr_dir + "/s4d_C.dat.npy").flatten().flatten()
    return s4d_A, s4d_B, s4d_C


def run_original_model(
        input: np.ndarray,
        num_steps: int,
        model_dim: int,
        d_states: int,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run original S4d model.

    Parameters
    ----------
    input: np.ndarray
        Input signal to the model.
    num_steps: int
        Number of time steps to simulate the model.
    model_dim: int
        Dimensionality of the model.
    d_states: int
        Number of model states.
    A: np.ndarray
        Diagonal elements of the state matrix of the S4D model.
    B: np.ndarray
        Diagonal elements of the input matrix of the S4D model.
    C: np.ndarray
        Diagonal elements of the output matrix of the S4D model.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple containing the output of the model simulation.

    Notes
    -----
    This function simulates the behavior of a linear time-invariant system
    with diagonalized state-space representation.
    The state-space equations are given by:
    x_{k+1} = A * x_k + B * u_k
    y_k = C * x_k

    where:
    - x_k is the state vector at time step k,
    - u_k is the input vector at time step k,
    - y_k is the output vector at time step k,
    - A is the diagonal state matrix,
    - B is the diagonal input matrix,
    - C is the diagonal output matrix.

    The function computes the output of the system for the given input signal
    over num_steps time steps.
    """

    A = A[:model_dim * d_states]
    B = B[:model_dim * d_states]
    C = C[:model_dim * d_states]
    expansion_weights = np.kron(np.eye(model_dim), np.ones(d_states))
    expanded_inp = np.matmul(expansion_weights.T, input)
    out = np.zeros((model_dim * d_states, num_steps))
    S4state = np.zeros((model_dim * d_states,)).flatten()

    for idx, inp in enumerate(expanded_inp.T):
        S4state = np.multiply(S4state, A) + np.multiply(inp, B)
        out[:, idx] = np.multiply(C, S4state) * 2

    out = np.matmul(expansion_weights, out)
    return out
