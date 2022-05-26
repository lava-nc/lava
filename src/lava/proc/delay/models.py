# Modified from src/lava/proc/dense/process.py: PyDenseModelFloat
# Modified by Kevin Sargent, Pennsylvania State University 

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.delay.process import Delay

@implements(proc=Delay, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDelayModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with dense synaptic connections in
    floating point precision. Modified from the Dense process to support synaptic delays.
    This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_buff: np.ndarray = LavaPyType(np.ndarray, bool)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    # weight_exp: float = LavaPyType(float, float)
    # num_weight_bits: float = LavaPyType(float, float)
    # sign_mode: float = LavaPyType(float, float)
    use_graded_spike: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    delays: np.ndarray = LavaPyType(np.ndarray, int)
    max_delay: int = LavaPyType(int, int)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        a_out = np.sum( np.take_along_axis(self.s_buff.T, self.delays, axis=0)*self.weights, axis=1)
        self.a_out.send(a_out)
        self.s_buff = np.roll(self.s_buff, 1, axis=1)
        if self.use_graded_spike.item():
            self.s_buff[:,0] = self.s_in.recv()
        else:
            self.s_buff[:,0] = self.s_in.recv().astype(bool)
