import numpy as np
import typing as ty
import numpy.typing as npty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var

class SpikeIntegrator(AbstractProcess):
    """GradedVec
    Graded spike vector layer. Accumulates and forwards 32bit spikes.
    Parameters
    ----------
    shape: tuple(int)
        number and topology of neurons
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            relu_flag = False,
            q_scale = 1,
            q_exp = 12, 
            state_exp = 0) -> None:
        super().__init__(shape=shape, relu_flag=relu_flag, state_exp=state_exp, q_scale=q_scale, q_exp=q_exp) #, log_config=log_config)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.data = Var(shape=shape, init=0)
        self.relu_flag = Var(shape=shape, init=int(relu_flag))
    


class SpikeIntegrator32(AbstractProcess):
    """GradedVec
    Graded spike vector layer. Accumulates and forwards 32bit spikes.
    Parameters
    ----------
    shape: tuple(int)
        number and topology of neurons
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            relu_flag = False, 
            state_exp = 0) -> None:
        super().__init__(shape=shape, relu_flag=relu_flag, state_exp=state_exp) #, log_config=log_config)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.data = Var(shape=shape, init=0)
        self.relu_flag = Var(shape=shape, init=int(relu_flag))
