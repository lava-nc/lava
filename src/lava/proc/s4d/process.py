# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.sdn.process import ActivationMode, SigmaDelta


class S4d(AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            a: float,
            b: float,
            c: float,
            s4_state: ty.Optional[int] = 0,
            s4_exp: ty.Optional[int] = 0,
            inp_exp: ty.Optional[int] = 0) -> None:
        """
        Neuron process that implements S4D (described by
        Gu et al., 2022) dynamics.

        This process simulates the behavior of a linear time-invariant system
        with diagonal state-space representation.
        The state-space equations are given by:
        s4_state_{k+1} = A * s4_state_k + B * inp_k
        act_k = C * s4_state_k

        where:
        - s4_state_k is the state vector at time step k,
        - inp_k is the input vector at time step k,
        - act_k is the output vector at time step k,
        - A is the diagonal state matrix,
        - B is the diagonal input matrix,
        - C is the diagonal output matrix.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        a: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        b: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        c: np.ndarray
            Diagonal elements of the output matrix of the S4D model.
        s4_state: int or float
            Initial state of the S4D model.
        s4_exp: int
            Scaling exponent with base 2 for the S4 state variables.
            Note: This should only be used for nc models.
            Default is 0.
        inp_exp: int
            Bit precision of the input signal.
            Note: This should only be used for nc models.
            Default is 0.
        """

        super().__init__(shape=shape,
                         a=a,
                         b=b,
                         c=c,
                         s4_state=s4_state,
                         s4_exp=s4_exp,
                         inp_exp=inp_exp)
        # Ports
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        # Variables for S4
        self.a = Var(shape=shape, init=a)
        self.b = Var(shape=shape, init=b)
        self.c = Var(shape=shape, init=c)
        self.s4_state = Var(shape=shape, init=s4_state)
        self.s4_exp = Var(shape=(1,), init=s4_exp)
        self.inp_exp = Var(shape=(1,), init=inp_exp)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class SigmaS4dDelta(SigmaDelta, AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Union[int, float],
            a: float,
            b: float,
            c: float,
            state_exp: ty.Optional[int] = 0,
            s4_exp: ty.Optional[int] = 0) -> None:
        """
        Sigma delta neuron process that implements S4D (described by
        Gu et al., 2022) dynamics as its activation function.

        This process simulates the behavior of a linear time-invariant system
        with diagonal state-space representation.
        The state-space equations are given by:
        s4_state_{k+1} = A * s4_state_k + B * inp_k
        act_k = C * s4_state_k

        where:
        - s4_state_k is the state vector at time step k,
        - inp_k is the input vector at time step k,
        - act_k is the output vector at time step k,
        - A is the diagonal state matrix,
        - B is the diagonal input matrix,
        - C is the diagonal output matrix.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        a: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        b: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        c: np.ndarray
            Diagonal elements of the output matrix of the S4D model.
        state_exp: int
            Scaling exponent with base 2 for the reconstructed sigma variables.
            Note: This should only be used for nc models.
            Default is 0.
        s4_exp: int
            Scaling exponent with base 2 for the S4 state variables.
            Note: This should only be used for nc models.
            Default is 0.
        """

        super().__init__(shape=shape,
                         vth=vth,
                         a=a,
                         b=b,
                         c=c,
                         s4_state=0,
                         state_exp=state_exp,
                         s4_exp=s4_exp)

        # Variables for S4
        self.a = Var(shape=shape, init=a)
        self.b = Var(shape=shape, init=b)
        self.c = Var(shape=shape, init=c)
        self.s4_state = Var(shape=shape, init=0)
        self.s4_exp = Var(shape=(1,), init=s4_exp)


class SigmaS4dDeltaLayer(AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Union[int, float],
            a: float,
            b: float,
            c: float,
            d_states: ty.Optional[int] = 1,
            s4_exp: ty.Optional[int] = 0,
            num_message_bits: ty.Optional[int] = 24,
            state_exp: ty.Optional[int] = 0) -> None:
        """
        Combines S4D neuron with Sparse Processes that allow for multiple
        d_states.

        Connectivity: Sparse -> SigmaS4dDelta -> Sparse.
        Relieves user from computing required connection weights for multiple
        d_states.

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        a: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        b: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        c: np.ndarray
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
        s4_exp: int
            Scaling exponent with base 2 for the S4 state variables.
            Note: Only relevant for nc model.
            Default is 0.
        """

        # Automatically takes care of expansion and reduction of dimensionality
        # for multiple hidden states (d_states)
        conn_weights = np.kron(np.eye(shape[0]), np.ones(d_states))
        s4_state = 0
        super().__init__(shape=shape,
                         vth=vth,
                         a=a,
                         b=b,
                         c=c,
                         s4_exp=s4_exp,
                         s4_state=s4_state,
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
        self.a = Var(shape=(shape[0] * d_states,), init=a)
        self.b = Var(shape=(shape[0] * d_states,), init=b)
        self.c = Var(shape=(shape[0] * d_states,), init=c)
        self.s4_state = Var(shape=(shape[0] * d_states,), init=0)
        self.S4_exp = Var(shape=(1,), init=s4_exp)

        # Variables for connecting Dense processes
        # Project input_dim to input_dim * d_states
        self.conn_weights = Var(shape=shape, init=conn_weights)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
