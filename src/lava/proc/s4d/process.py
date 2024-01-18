# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.sdn.process import ActivationMode, SigmaDelta
import typing as ty
import numpy as np


class SigmaS4Delta(SigmaDelta, AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Union[int, float],
            A: float,
            B: float,
            C: float,
            state_exp: ty.Optional[int] = 0,
            S4_exp: ty.Optional[int] = 0) -> None:
        """
        Sigma delta neuron process that implements S4D dynamics as its
        activation function.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        A: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        B: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        C: np.ndarray
            Diagonal elements of the output matrix of the S4D model.
        state_exp: int
            Scaling exponent with base 2 for the reconstructed sigma variables.
            Note: This should only be used for nc models.
            Default is 0.
        S4_exp: int
            Scaling exponent with base 2 for the S4 state variables.
            Note: This should only be used for nc models.
            Default is 0.

        Notes
        -----
        This process simulates the behavior of a linear time-invariant system
        with diagonal state-space representation.
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
        """

        super().__init__(shape=shape,
                         vth=vth,
                         A=A,
                         B=B,
                         C=C,
                         S4state=0,
                         state_exp=state_exp,
                         S4_exp=S4_exp)

        # Variables for S4
        self.A = Var(shape=shape, init=A)
        self.B = Var(shape=shape, init=B)
        self.C = Var(shape=shape, init=C)
        self.S4state = Var(shape=shape, init=0)
        self.S4_exp = Var(shape=(1,), init=S4_exp)


class SigmaS4DeltaLayer(AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Union[int, float],
            A: float,
            B: float,
            C: float,
            d_states: ty.Optional[int] = 1,
            S4_exp: ty.Optional[int] = 0,
            num_message_bits: ty.Optional[int] = 24,
            state_exp: ty.Optional[int] = 0) -> None:
        """
        Combines S4D neuron with Sparse Processes that allow for multiple
        d_states.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        A: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        B: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        C: np.ndarray
            Diagonal elements of the output matrix of the S4D model.
        d_states: int
            Number of hidden states of the S4D model.
            Default is 1.
        state_exp: int
            Scaling exponent with base 2 for the reconstructed sigma variables.
            Note: Only relevant for nc model.
            Default is 0.
        num_message_bits: int
            Number of message bits to be used in Sparse connection processes.
            Note: Only relevant for nc model.
        S4_exp: int
            Scaling exponent with base 2 for the S4 state variables.
            Note: Only relevant for nc model.
            Default is 0.

        Notes
        -----
        Connectivity: Sparse -> SigmaS4Delta -> Sparse.
        Relieves user from computing required connection weights for multiple
        d_states.

        This process simulates the behavior of a linear time-invariant system
        with diagonalized state-space representation. (S4D)
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
        """

        # Automatically takes care of expansion and reduction of dimensionality
        # for multiple hidden states (d_states)
        conn_weights = np.kron(np.eye(shape[0]), np.ones(d_states))
        S4state = 0
        super().__init__(shape=shape,
                         vth=vth,
                         A=A,
                         B=B,
                         C=C,
                         S4_exp=S4_exp,
                         S4state=S4state,
                         conn_weights=conn_weights,
                         num_message_bits=num_message_bits,
                         d_states=d_states,
                         state_exp=state_exp,
                         act_mode=ActivationMode.UNIT)

        # Ports
        self.s_in = InPort(shape=shape)
        self.a_out = OutPort(shape=shape)

        # General variables
        self.state_exp = Var(shape=(1,), init=state_exp)

        # Variables for S4
        self.A = Var(shape=(shape[0] * d_states,), init=A)
        self.B = Var(shape=(shape[0] * d_states,), init=B)
        self.C = Var(shape=(shape[0] * d_states,), init=C)
        self.S4state = Var(shape=(shape[0] * d_states,), init=0)
        self.S4_exp = Var(shape=(1,), init=S4_exp)

        # Variables for connecting Dense processes
        # Project input_dim to input_dim * d_states
        self.conn_weights = Var(shape=shape, init=conn_weights)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
