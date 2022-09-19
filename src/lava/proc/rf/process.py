"""Code based on the tutorial for now:
https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial02_processes.ipynb

"""
import numpy as np
import typing as ty
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractRF(AbstractProcess):
    def __init__(self, 
            shape: ty.Tuple[int, ...],
            sin_decay: float,
            cos_decay: float, 
            vth: float,
            name: str,
            log_config: LogConfig) -> None:
        super().__init__(shape=shape,sin_decay=sin_decay,cos_decay=cos_decay,vth=vth,name=name,log_config=log_config)
        
        self.a_real_in = InPort(shape=shape)
        self.a_imag_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.real = Var(shape=shape, init=0)  # seems to force this value and we can't set it initally in constructor
        self.imag = Var(shape=shape, init=0)
        self.sin_decay = Var(shape=(1,), init=sin_decay)
        self.cos_decay = Var(shape=(1,), init=cos_decay)
        self.vth = Var(shape=(1,), init=vth)  # why is this 10

    # def print_vars(self):
    #     """Prints all variables of a LIF process and their values."""

    #     sp = 3 * "  "
    #     print("Variables of the LIF:")
    #     print(sp + "sin_decay:    {}".format(str(self.sin_decay.get())))
    #     print(sp + "cos_decay:    {}".format(str(self.cos_decay.get())))
    #     print(sp + "vth:   {}".format(str(self.vth.get())))
    #     print(sp + "real:   {}".format(str(self.real.get())))
    #     print(sp + "imag: {}".format(str(self.imag.get())))


class RF(AbstractRF):
    def __init__(self,
        shape: ty.Tuple[int, ...],
        sin_decay: ty.Optional[float] = 0,
        cos_decay: ty.Optional[float] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None
    ):
        super().__init__(shape=shape,sin_decay=sin_decay,cos_decay=cos_decay,vth=vth,name=name,log_config=log_config)

        self.vth = Var(shape=(1,), init=vth)