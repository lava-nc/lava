import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.rf.process import RF
from copy import deepcopy

class AbstractPyRFModelFloat(PyLoihiProcessModel):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    real: np.ndarray = LavaPyType(np.ndarray, float)
    imag: np.ndarray = LavaPyType(np.ndarray, float)
    sin_decay: float = LavaPyType(float, float)
    cos_decay: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def sub_thresh_dynamics(self):

        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()

        old_real = deepcopy(self.real)
        old_imag = deepcopy(self.imag)
        self.real = self.cos_decay * old_real - self.sin_decay * old_imag + a_real_in_data  # I don't think it needs a bias for now
        self.imag = self.sin_decay * old_real + self.cos_decay * old_imag + a_imag_in_data

    def run_spk(self):
        self.sub_thresh_dynamics()
        s_out = self.real >= self.vth 
        
        self.s_out.send(s_out)




@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRFModelFloat(AbstractPyRFModelFloat):
    pass




