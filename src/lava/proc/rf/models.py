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
        old_imag = deepcopy(self.imag)
        self.sub_thresh_dynamics()
        s_out = (self.real >= self.vth)  * (self.imag >= 0) * (old_imag < 0)
        self.s_out.send(s_out)




@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRFModelFloat(AbstractPyRFModelFloat):
    pass

class AbstractPyRFModelFixed(PyLoihiProcessModel):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: None
    real: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    imag: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sin_decay: int = LavaPyType(int, np.uint16, precision=12)   # 0 4095
    cos_decay:int = LavaPyType(int, np.uint16, precision=12)   # 0 to 4095
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def __init__(self, proc_params):
        super(AbstractPyRFModelFixed, self).__init__(proc_params)
        
        self.decay_shift = 12
        self.ri_bitwidth = 24 # real, imaginary bitwidth
        self.max_ri_val = 2 ** (self.ri_bitwidth - 1)
        self.act_shift = 6
        self.vth_shift = 6

        self.c_offset = 1
        self.s_offset = 1

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)

    def sub_thresh_dynamics(self):

        decay_const_cos = self.cos_decay + self.c_offset
        decay_const_sin = self.sin_decay + self.s_offset

        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()


        decayed_real = np.int64(self.real) * (decay_const_cos) - np.int64(self.imag) * (decay_const_sin)
        decayed_real = np.sign(decayed_real) * np.right_shift(np.abs(decayed_real), self.decay_shift)
        decayed_real = np.int32(decayed_real)

        a_real_in_data = np.left_shift(a_real_in_data, self.act_shift)
        decayed_real += a_real_in_data

        decayed_imag = np.int64(self.real) * ( decay_const_sin) + np.int64(self.imag) * (decay_const_cos)
        decayed_imag = np.sign(decayed_imag) * np.right_shift(np.abs(decayed_imag), self.decay_shift)
        decayed_imag = np.int32(decayed_imag)
        
        a_imag_in_data = np.left_shift(a_imag_in_data, self.act_shift)
        decayed_imag += a_imag_in_data

        neg_voltage_limit = -np.int32(self.max_ri_val) + 1
        pos_voltage_limit = np.int32(self.max_ri_val) - 1

        self.real[:] = np.clip(decayed_real, neg_voltage_limit, pos_voltage_limit)
        self.imag[:] = np.clip(decayed_imag, neg_voltage_limit, pos_voltage_limit)


    def run_spk(self):
        old_imag = deepcopy(self.imag)
        self.scale_threshold()
        self.sub_thresh_dynamics()

        s_out = (self.real >= self.effective_vth) *  (self.imag >= 0) * (old_imag < 0)  # prevents spiking twice in a row
        self.s_out.send(s_out)

@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('bit_accurate_loihi', 'fixed_pt')
class PyRFModelBitAcc(AbstractPyRFModelFixed):

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyRFModelBitAcc, self).__init__(proc_params)
