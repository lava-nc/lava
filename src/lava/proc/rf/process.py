"""Code based on the tutorial for now:
https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial02_processes.ipynb

"""
import typing as ty
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class RF(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 sin_decay: ty.Optional[float],
                 cos_decay: ty.Optional[float],
                 vth: ty.Optional[float] = 1,
                 state_exp: ty.Optional[int] = 0,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None):
        super().__init__(shape=shape, sin_decay=sin_decay, cos_decay=cos_decay,
                         vth=vth, state_exp=state_exp, name=name,
                         log_config=log_config)

        vth = vth * (1 << state_exp)
        self.a_real_in = InPort(shape=shape)
        self.a_imag_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.real = Var(shape=shape, init=0)
        self.imag = Var(shape=shape, init=0)
        self.sin_decay = Var(shape=(1,), init=sin_decay)
        self.cos_decay = Var(shape=(1,), init=cos_decay)
        self.vth = Var(shape=(1,), init=vth)
        self.state_exp = Var(shape=(1,), init=state_exp)
