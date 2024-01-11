from lava.magma.core.process.process import  AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import typing as ty



class SigmaS4Delta(AbstractProcess):
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            vth: float,
            A: float,
            B: float,
            C: float,
            state_exp: ty.Optional[int] = 0) -> None:
        """Sigma delta neuron process. That has S4d as its activation function"""
        
        self.S4state = Var(shape=shape, init=0)
        super().__init__(shape=shape, vth=vth, A=A, B=B, C=C, S4state=self.S4state, state_exp=state_exp)
        # scaling factor for fixed precision scaling
        vth = vth * (1 << (state_exp))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        
        # Variables for SigmaDelta
        self.vth = Var(shape=(1,), init=vth)
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.state_exp = Var(shape=(1,), init=state_exp)
        
        # Variables for S4        
        self.A = Var(shape=shape, init=A)
        self.B = Var(shape=shape, init=B)
        self.C = Var(shape=shape, init=C)
        

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']