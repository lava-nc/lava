# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt
import typing as ty
import numpy as np

from lava.proc.lif.process import LIF, AbstractLIF, LogConfig
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import LearningDense, Dense
from lava.magma.core.process.neuron import LearningNeuronProcess
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.neuron import LearningNeuronModelFloat
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFloat


class RSTDPLIF(LearningNeuronProcess, AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with RSTDP learning rule.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.

    """
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            du: ty.Optional[float] = 0,
            dv: ty.Optional[float] = 0,
            bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            vth: ty.Optional[float] = 10,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
            **kwargs) -> None:
        super().__init__(shape=shape, u=u, v=v, du=du, dv=dv,
                         bias_mant=bias_mant,
                         bias_exp=bias_exp, name=name,
                         log_config=log_config,
                         **kwargs)
        self.vth = Var(shape=(1,), init=vth)

        self.a_graded_reward_in = InPort(shape=shape)


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class RSTDPLIFModel(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in floating point precision with learning enabled.
    """
    # Graded reward input spikes
    a_graded_reward_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params['shape'])

    def spiking_activation(self):
        """Spiking activation function for Learning LIF.
        """
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor Reward traces based on
        Graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """
        super().run_spk()

        a_graded_in = self.a_graded_reward_in.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_y1.send(self.s_out_buff)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


def generate_post_spikes(pre_spike_times, 
        num_steps, spike_prob_post):
    """generates specific post synaptic spikes to
    demonstrate potentiation and depression.
    """
    pre_synaptic_spikes = np.where(pre_spike_times==1)[1]

    spike_raster_post = np.zeros((len(spike_prob_post), num_steps))

    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts, pre_ts+20):
                if np.random.rand(1) < spike_prob_post[0]:
                    spike_raster_post[0][ts] = 1

    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts-12, pre_ts-2):
                if np.random.rand(1) < spike_prob_post[1]:
                    spike_raster_post[1][ts] = 1
    
    return spike_raster_post

def plot_spikes(spikes, figsize, legend, colors, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps+1, 25)
    
    plt.figure(figsize=figsize)

    spikes_plot = plt.eventplot(positions=spikes, 
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")

    plt.xticks(num_x_ticks)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    
    plt.yticks(ticks=offsets, labels=legend)

    
    plt.show()

def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    plt.step(time, time_series, color=color)
   
    plt.title(title)
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.ylabel(ylabel)
    
    plt.show()

def plot_time_series_subplots(time, time_series_y1, time_series_y2, ylabel, title, figsize, color, legend, leg_loc="upper left"):    
    plt.figure(figsize=figsize)
    
    plt.step(time, time_series_y1, label=legend[0], color=color[0])
    plt.step(time, time_series_y2, label=legend[1], color=color[1])
        
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.xlim(0, len(time_series_y1))

    plt.legend(loc=leg_loc)
    
    plt.show()

def plot_spikes_time_series(time, time_series, spikes, figsize, legend, colors, title, num_steps):

    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps+1, 25)
    
    plt.figure(figsize=figsize)
    
    plt.subplot(211)
    plt.eventplot(positions=spikes, 
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors)

    plt.title("Spike Arrival")
    plt.xlabel("Time steps")

    plt.xticks(num_x_ticks)
    plt.xlim(0, num_steps)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    
    plt.yticks(ticks=offsets, labels=legend)
    plt.tight_layout(pad=3.0)

    plt.subplot(212)
    plt.step(time, time_series, color=colors)
   
    plt.title(title[0])
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.margins(x=0)

    plt.ylabel("Trace Value")
    
    plt.show()
