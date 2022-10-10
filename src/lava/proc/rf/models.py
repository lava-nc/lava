import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.rf.process import RF


class AbstractPyRFModelFloat(PyLoihiProcessModel):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    real: np.ndarray = LavaPyType(np.ndarray, float)
    imag: np.ndarray = LavaPyType(np.ndarray, float)
    sin_decay: float = LavaPyType(float, float)
    cos_decay: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)

    def sub_thresh_dynamics(self):

        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()

        # I don't think it needs a bias for now
        decayed_real = (self.cos_decay * self.real -
                        self.sin_decay * self.imag + a_real_in_data)
        decayed_imag = (self.sin_decay * self.real +
                        self.cos_decay * self.imag + a_imag_in_data)

        self.real[:] = decayed_real
        self.imag[:] = decayed_imag

    def run_spk(self):
        old_imag = self.imag.copy()
        self.sub_thresh_dynamics()
        s_out = (self.real >= self.vth) * (self.imag >= 0) * (old_imag < 0)
        self.s_out.send(s_out)


@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRFModelFloat(AbstractPyRFModelFloat):
    pass


class AbstractPyRFModelFixed(PyLoihiProcessModel):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE,
                                     np.int16, precision=16)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, 
                                     np.int16, precision=16)
    s_out: None
    real: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    imag: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sin_decay: int = LavaPyType(int, np.uint16, precision=12)
    cos_decay: int = LavaPyType(int, np.uint16, precision=12)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)

    def __init__(self, proc_params):
        super(AbstractPyRFModelFixed, self).__init__(proc_params)

        self.decay_shift = self.sin_decay.precision

        # real, imaginary bitwidth
        self.ri_bitwidth = self.sin_decay.precision * 2
        max_ri_val = 2 ** (self.ri_bitwidth - 1)
        self.neg_voltage_limit = -np.int32(max_ri_val) + 1
        self.pos_voltage_limit = np.int32(max_ri_val) - 1

    def sub_thresh_dynamics(self):

        decay_const_cos = self.cos_decay
        decay_const_sin = self.sin_decay

        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()

        decayed_real = (np.int64(self.real) * (decay_const_cos)
                        - np.int64(self.imag) * (decay_const_sin))
        decayed_real = (np.sign(decayed_real) *
                        np.right_shift(np.abs(decayed_real), self.decay_shift))
        decayed_real = np.int32(decayed_real)

        a_real_in_data = np.left_shift(a_real_in_data, self.state_exp)
        decayed_real += a_real_in_data

        decayed_imag = (np.int64(self.real) * (decay_const_sin)
                        + np.int64(self.imag) * (decay_const_cos))
        decayed_imag = (np.sign(decayed_imag) *
                        np.right_shift(np.abs(decayed_imag), self.decay_shift))
        decayed_imag = np.int32(decayed_imag)

        a_imag_in_data = np.left_shift(a_imag_in_data, self.state_exp)
        decayed_imag += a_imag_in_data

        self.real[:] = np.clip(decayed_real,
                               self.neg_voltage_limit, self.pos_voltage_limit)
        self.imag[:] = np.clip(decayed_imag, 
                               self.neg_voltage_limit, self.pos_voltage_limit)

    def run_spk(self):
        old_imag = self.imag.copy()
        self.sub_thresh_dynamics()
        s_out = (self.real >= self.vth) * (self.imag >= 0) * (old_imag < 0)
        self.s_out.send(s_out)


@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('bit_accurate_loihi', 'fixed_pt')
class PyRFModelBitAcc(AbstractPyRFModelFixed):

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyRFModelBitAcc, self).__init__(proc_params)
