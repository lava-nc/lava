from lava.proc.sdn.models import AbstractSigmaModel, AbstractDeltaModel
import typing as ty
from typing import Any, Dict
from lava.magma.core.decorator import implements, requires, tag
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.s4d.process import SigmaS4Delta
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

class AbstractSigmaS4DeltaModel(AbstractSigmaModel, AbstractDeltaModel):
    a_in = None
    s_out = None

    vth = None
    sigma = None
    act = None
    residue = None
    error = None
    state_exp = None
    
    A = None
    B = None
    C = None
    S4state = None
    

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        super().__init__(proc_params)
        self.A = self.proc_params['A']
        self.B = self.proc_params['B']
        self.C = self.proc_params['C']
        self.S4state = self.proc_params['S4state']

    def activation_dynamics(self, sigma_data: np.ndarray) -> np.ndarray:
        """Sigma Delta activation dynamics. UNIT and RELU activations are
        supported.

        Parameters
        ----------
        sigma_data : np.ndarray
            sigma decoded data

        Returns
        -------
        np.ndarray
            activation output
        
        
        Equations: 
        initial state = model.default_state (apparently all zeros)
        new_state = state * A + inp * B
        out = c * new_state * 2 # mal zwei remains unclear
        """
        self.S4state = self.S4state * self.A + sigma_data * self.B
        act = self.C * self.S4state * 2
        return act

    def dynamics(self, a_in_data: np.ndarray) -> np.ndarray:
        self.sigma = self.sigma_dynamics(a_in_data)
        act = self.activation_dynamics(self.sigma)
        s_out = self.delta_dynamics(act)
        self.act = act
        return s_out


@implements(proc=SigmaS4Delta, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySigmaS4DeltaModelFloat(AbstractSigmaS4DeltaModel):
    """Floating point implementation of Sigma Delta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)

    vth: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: np.ndarray = LavaPyType(np.ndarray, float)
    act: np.ndarray = LavaPyType(np.ndarray, float)
    residue: np.ndarray = LavaPyType(np.ndarray, float)
    error: np.ndarray = LavaPyType(np.ndarray, float)
    
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
  
        
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