# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import typing as ty
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class Plateau(AbstractProcess):
    """Plateau Neuron Process.

    Couples two modified LIF dynamics. The neuron posesses two voltages,
    v_dend and v_soma. Both follow sub-threshold LIF dynamics. When v_dend
    crosses v_th_dend, it resets and sets the up_state to the value up_dur.
    The supra-threshold behavior of v_soma depends on up_state:
        if up_state == 0:
            v_soma follows sub-threshold dynamics
        if up_state > 0:
            v_soma resets and the neuron sends out a spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of Plateau neurons.
    dv_dend : int
        Inverse of the decay time-constant for the dendrite voltage.
    dv_soma : int
        Inverse of the decay time-constant for the soma voltage.
    vth_dend : int
        Dendrite threshold voltage, exceeding which, the neuron will enter the
        UP state.
    vth_soma : int
        Soma threshold voltage, exceeding which, the neuron will spike if it is
        also in the UP state.
    up_dur : int
        The duration, in timesteps, of the UP state.
    """
    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        dv_dend: int,
        dv_soma: int,
        vth_dend: int,
        vth_soma: int,
        up_dur: int,
        name: ty.Optional[str] = None,
    ):
        super().__init__(
            shape=shape,
            dv_dend=dv_dend,
            dv_soma=dv_soma,
            name=name,
            up_dur=up_dur,
            vth_dend=vth_dend,
            vth_soma=vth_soma
        )
        self.a_dend_in = InPort(shape=shape)
        self.a_soma_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v_dend = Var(shape=shape, init=0)
        self.v_soma = Var(shape=shape, init=0)
        self.dv_dend = Var(shape=(1,), init=dv_dend)
        self.dv_soma = Var(shape=(1,), init=dv_soma)
        self.vth_dend = Var(shape=(1,), init=vth_dend)
        self.vth_soma = Var(shape=(1,), init=vth_soma)
        self.up_dur = Var(shape=(1,), init=up_dur)
        self.up_state = Var(shape=shape, init=0)
