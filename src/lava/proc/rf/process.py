"""Code based on the tutorial for now:
https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial02_processes.ipynb

"""
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

class AbstractRF(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.a_real_in = InPort(shape=shape)
        self.a_imag_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.real = Var(shape=shape, init=0)  # seems to force this value and we can't set it initally in constructor
        self.imag = Var(shape=shape, init=0)
        self.sin_decay = Var(shape=(1,), init=kwargs.pop("sin_decay", 0))
        self.cos_decay = Var(shape=(1,), init=kwargs.pop("cos_decay", 0))
        self.vth = Var(shape=(1,), init=kwargs.pop("vth", 10))  # why is this 10

    def print_vars(self):
        """Prints all variables of a LIF process and their values."""

        sp = 3 * "  "
        print("Variables of the LIF:")
        print(sp + "sin_decay:    {}".format(str(self.sin_decay.get())))
        print(sp + "cos_decay:    {}".format(str(self.cos_decay.get())))
        print(sp + "vth:   {}".format(str(self.vth.get())))
        print(sp + "real:   {}".format(str(self.real.get())))
        print(sp + "imag: {}".format(str(self.imag.get())))


class RF(AbstractRF):
    pass