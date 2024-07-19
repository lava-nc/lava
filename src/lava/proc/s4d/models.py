# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from typing import Any, Dict
from lava.proc.sdn.models import AbstractSigmaDeltaModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.s4d.process import SigmaS4dDelta, SigmaS4dDeltaLayer, S4d
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.sparse.process import Sparse
from lava.magma.core.model.py.model import PyLoihiProcessModel


@implements(proc=S4d, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class S4dModel(PyLoihiProcessModel):
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)
    s4_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    inp_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)

    # S4 variables
    s4_state: np.ndarray = LavaPyType(np.ndarray, complex)
    a: np.ndarray = LavaPyType(np.ndarray, complex)
    b: np.ndarray = LavaPyType(np.ndarray, complex)
    c: np.ndarray = LavaPyType(np.ndarray, complex)

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        """
        Neuron model that implements S4D
        (as described by Gu et al., 2022) dynamics.

        Relevant parameters in proc_params
        --------------------------
        a: np.ndarray
            Diagonal elements of the state matrix of the discretized S4D model.
        b: np.ndarray
            Diagonal elements of the input matrix of the discretized S4D model.
        c: np.ndarray
            Diagonal elements of the output matrix of the discretized S4D model.
        s4_state: np.ndarray
            State vector of the S4D discretized model.
        """
        super().__init__(proc_params)
        self.a = self.proc_params['a']
        self.b = self.proc_params['b']
        self.c = self.proc_params['c']
        self.s4_state = self.proc_params['s4_state']

    def run_spk(self) -> None:
        """Performs S4D dynamics.

        This function simulates the behavior of a linear time-invariant system
        with diagonalized state-space representation.
        (For reference see Gu et al., 2022)

        The state-space equations are given by:
        s4_state_{k+1} = A * s4_state_k + B * input_k
        act_k = C * s4_state_k

        where:
        - s4_state_k is the state vector at time step k,
        - input_k is the input vector at time step k,
        - act_k is the output vector at time step k,
        - A is the diagonal state matrix,
        - B is the diagonal input matrix,
        - C is the diagonal output matrix.

        The function computes the next output step of the
        system for the given input signal.
        """
        inp = self.a_in.recv()
        self.s4_state = (self.s4_state * self.a + inp * self.b)
        self.s_out.send(np.real(self.c * self.s4_state * 2))


class AbstractSigmaS4dDeltaModel(AbstractSigmaDeltaModel):
    a_in = None
    s_out = None

    # SigmaDelta Variables
    vth = None
    sigma = None
    act = None
    residue = None
    error = None
    state_exp = None
    bias = None

    # S4 Variables
    a = None
    b = None
    c = None
    s4_state = None
    s4_exp = None

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        """
        Sigma delta neuron model that implements S4D
        (as described by Gu et al., 2022) dynamics as its activation function.

        Relevant parameters in proc_params
        --------------------------
        a: np.ndarray
            Diagonal elements of the state matrix of the S4D model.
        b: np.ndarray
            Diagonal elements of the input matrix of the S4D model.
        c: np.ndarray
            Diagonal elements of the output matrix of the S4D model.
        s4_state: np.ndarray
            State vector of the S4D model.
        """
        super().__init__(proc_params)
        self.a = self.proc_params['a']
        self.b = self.proc_params['b']
        self.c = self.proc_params['c']
        self.s4_state = self.proc_params['s4_state']

    def activation_dynamics(self, sigma_data: np.ndarray) -> np.ndarray:
        """Sigma Delta activation dynamics. Performs S4D dynamics.

        This function simulates the behavior of a linear time-invariant system
        with diagonalized state-space representation.
        (For reference see Gu et al., 2022)

        The state-space equations are given by:
        s4_state_{k+1} = A * s4_state_k + B * input_k
        act_k = C * s4_state_k

        where:
        - s4_state_k is the state vector at time step k,
        - input_k is the input vector at time step k,
        - act_k is the output vector at time step k,
        - A is the diagonal state matrix,
        - B is the diagonal input matrix,
        - C is the diagonal output matrix.

        The function computes the next output step of the
        system for the given input signal.

        Parameters
        ----------
        sigma_data: np.ndarray
            sigma decoded data

        Returns
        -------
        np.ndarray
            activation output
        """

        self.s4_state = self.s4_state * self.a + sigma_data * self.b
        act = self.c * self.s4_state * 2
        return act


@implements(proc=SigmaS4dDelta, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySigmaS4dDeltaModelFloat(AbstractSigmaS4dDeltaModel):
    """Floating point implementation of SigmaS4dDelta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)

    vth: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: np.ndarray = LavaPyType(np.ndarray, float)
    act: np.ndarray = LavaPyType(np.ndarray, float)
    residue: np.ndarray = LavaPyType(np.ndarray, float)
    error: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)

    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    s4_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)

    # S4 vaiables
    s4_state: np.ndarray = LavaPyType(np.ndarray, float)
    a: np.ndarray = LavaPyType(np.ndarray, float)
    b: np.ndarray = LavaPyType(np.ndarray, float)
    c: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        s_out = self.dynamics(a_in_data)
        self.s_out.send(s_out)


@implements(proc=SigmaS4dDeltaLayer, protocol=LoihiProtocol)
class SubDenseLayerModel(AbstractSubProcessModel):
    def __init__(self, proc):
        """Builds (Sparse -> S4D -> Sparse) connection of the process."""
        conn_weights = proc.proc_params.get("conn_weights")
        shape = proc.proc_params.get("shape")
        state_exp = proc.proc_params.get("state_exp")
        num_message_bits = proc.proc_params.get("num_message_bits")
        s4_exp = proc.proc_params.get("s4_exp")
        d_states = proc.proc_params.get("d_states")
        a = proc.proc_params.get("a")
        b = proc.proc_params.get("b")
        c = proc.proc_params.get("c")
        vth = proc.proc_params.get("vth")

        # Instantiate processes
        self.sparse1 = Sparse(weights=conn_weights.T, weight_exp=state_exp,
                              num_message_bits=num_message_bits)
        self.sigma_S4d_delta = SigmaS4dDelta(shape=(shape[0] * d_states,),
                                             vth=vth,
                                             state_exp=state_exp,
                                             s4_exp=s4_exp,
                                             a=a,
                                             b=b,
                                             c=c)
        self.sparse2 = Sparse(weights=conn_weights, weight_exp=state_exp,
                              num_message_bits=num_message_bits)

        # Make connections Sparse -> SigmaS4Delta -> Sparse
        proc.in_ports.s_in.connect(self.sparse1.in_ports.s_in)
        self.sparse1.out_ports.a_out.connect(self.sigma_S4d_delta.in_ports.a_in)
        self.sigma_S4d_delta.out_ports.s_out.connect(self.sparse2.s_in)
        self.sparse2.out_ports.a_out.connect(proc.out_ports.a_out)

        # Set aliases
        proc.vars.a.alias(self.sigma_S4d_delta.vars.a)
        proc.vars.b.alias(self.sigma_S4d_delta.vars.b)
        proc.vars.c.alias(self.sigma_S4d_delta.vars.c)
        proc.vars.s4_state.alias(self.sigma_S4d_delta.vars.s4_state)
