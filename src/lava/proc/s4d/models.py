# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.proc.sdn.models import AbstractSigmaDeltaModel
from typing import Any, Dict
from lava.magma.core.decorator import implements, requires, tag
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.s4d.process import SigmaS4Delta, SigmaS4DeltaLayer
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.sparse.process import Sparse


class AbstractSigmaS4DeltaModel(AbstractSigmaDeltaModel):
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
    A = None
    B = None
    C = None
    S4state = None
    S4_exp = None

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        super().__init__(proc_params)
        self.A = self.proc_params['A']
        self.B = self.proc_params['B']
        self.C = self.proc_params['C']
        self.S4state = self.proc_params['S4state']

    def activation_dynamics(self, sigma_data: np.ndarray) -> np.ndarray:
        """Sigma Delta activation dynamics. Performs S4D dynamics.

        Parameters
        ----------
        sigma_data: np.ndarray
            sigma decoded data

        Returns
        -------
        np.ndarray
            activation output

        Notes
        -----
        This function simulates the behavior of a linear time-invariant system
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

        The function computes the next output step of the
        system for the given input signal.
        """

        self.S4state = self.S4state * self.A + sigma_data * self.B
        act = self.C * self.S4state * 2
        return act


@implements(proc=SigmaS4Delta, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySigmaS4DeltaModelFloat(AbstractSigmaS4DeltaModel):
    """Floating point implementation of SigmaS4Delta neuron."""
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
    S4_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)

    # S4 stuff
    S4state: np.ndarray = LavaPyType(np.ndarray, float)
    A: np.ndarray = LavaPyType(np.ndarray, float)
    B: np.ndarray = LavaPyType(np.ndarray, float)
    C: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        s_out = self.dynamics(a_in_data)
        self.s_out.send(s_out)


@implements(proc=SigmaS4DeltaLayer, protocol=LoihiProtocol)
class SubDenseLayerModel(AbstractSubProcessModel):
    def __init__(self, proc):
        """Builds (Sparse -> S4D -> Sparse) connection of the process."""
        conn_weights = proc.proc_params.get("conn_weights")
        shape = proc.proc_params.get("shape")
        state_exp = proc.proc_params.get("state_exp")
        num_message_bits = proc.proc_params.get("num_message_bits")
        S4_exp = proc.proc_params.get("S4_exp")
        d_states = proc.proc_params.get("d_states")
        A = proc.proc_params.get("A")
        B = proc.proc_params.get("B")
        C = proc.proc_params.get("C")
        vth = proc.proc_params.get("vth")

        # Instantiate processes
        self.sparse1 = Sparse(weights=conn_weights.T, weight_exp=state_exp,
                              num_message_bits=num_message_bits)
        self.sigmaS4delta = SigmaS4Delta(shape=(shape[0] * d_states,),
                                         vth=vth,
                                         state_exp=state_exp,
                                         S4_exp=S4_exp,
                                         A=A,
                                         B=B,
                                         C=C)
        self.sparse2 = Sparse(weights=conn_weights, weight_exp=state_exp,
                              num_message_bits=num_message_bits)

        # Make connections Sparse -> SigmaS4Delta -> Sparse
        proc.in_ports.s_in.connect(self.sparse1.in_ports.s_in)
        self.sparse1.out_ports.a_out.connect(self.sigmaS4delta.in_ports.a_in)
        self.sigmaS4delta.out_ports.s_out.connect(self.sparse2.s_in)
        self.sparse2.out_ports.a_out.connect(proc.out_ports.a_out)

        # Set aliasses
        proc.vars.A.alias(self.sigmaS4delta.vars.A)
        proc.vars.B.alias(self.sigmaS4delta.vars.B)
        proc.vars.C.alias(self.sigmaS4delta.vars.C)
        proc.vars.S4state.alias(self.sigmaS4delta.vars.S4state)
