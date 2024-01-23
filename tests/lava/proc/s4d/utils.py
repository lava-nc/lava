# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import numpy as np
from typing import Tuple


def get_coefficients() -> [np.ndarray, np.ndarray, np.ndarray]:
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Initialize A, B and C with values trained on efficientnet features.
    s4d_A = np.load(curr_dir + "/s4d_A.dat.npy").flatten()
    s4d_B = np.load(curr_dir + "/s4d_B.dat.npy").flatten()
    s4d_C = np.load(curr_dir + "/s4d_C.dat.npy").flatten().flatten()
    return s4d_A, s4d_B, s4d_C


def run_original_model(
        inp: np.ndarray,
        num_steps: int,
        model_dim: int,
        d_states: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run original S4d model.

    This function simulates the behavior of a linear time-invariant system
    with diagonalized state-space representation. (S4D)
    The state-space equations are given by:
    s4_state_{k+1} = A * s4_state_k + B * input_k
    out_k = C * s4_state_k

    where:
    - s4_state_k is the state vector at time step k,
    - input_k is the input vector at time step k,
    - out_k is the output vector at time step k,
    - A is the diagonal state matrix,
    - B is the diagonal input matrix,
    - C is the diagonal output matrix.

    The function computes the next output step of the
    system for the given input signal.

    The function computes the output of the system for the given input signal
    over num_steps time steps.

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
    a: np.ndarray
        Diagonal elements of the state matrix of the S4D model.
    b: np.ndarray
        Diagonal elements of the input matrix of the S4D model.
    c: np.ndarray
        Diagonal elements of the output matrix of the S4D model.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple containing the output of the model simulation.
    """

    a = a[:model_dim * d_states]
    b = b[:model_dim * d_states]
    c = c[:model_dim * d_states]
    expansion_weights = np.kron(np.eye(model_dim), np.ones(d_states))
    expanded_inp = np.matmul(expansion_weights.T, inp)
    out = np.zeros((model_dim * d_states, num_steps))
    s4_state = np.zeros((model_dim * d_states,)).flatten()

    for idx, inp in enumerate(expanded_inp.T):
        s4_state = np.multiply(s4_state, a) + np.multiply(inp, b)
        out[:, idx] = np.multiply(c, s4_state) * 2

    out = np.matmul(expansion_weights, out)
    return out
