import matplotlib.pyplot as plt
import numpy as np
import typing as ty

from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyRefPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.neuron import LearningNeuronProcess
from lava.magma.core.model.py.neuron import LearningNeuronModelFloat
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

from lava.proc.lif.process import AbstractLIF
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.dense.process import LearningDense, Dense
from lava.proc.dense.models import PyLearningDenseModelFloat
from lava.proc.monitor.process import Monitor
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer as SpikeIn

####################################################################
# Creating a custom SuperSpikeLIF
####################################################################
class SuperSpikeLIF(LearningNeuronProcess, AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with learning enabled.

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
                         log_config=log_config, **kwargs)
        self.vth = Var(shape=(1,), init=vth)

        self.s_error_out = Var(shape=shape, init=np.zeros(shape))
        
        # Third factor input
        self.a_third_factor_in = InPort(shape=shape)

        self.v_port = OutPort(shape=(1,))


@implements(proc=SuperSpikeLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySuperSpikeLifModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision with learning enabled. 
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)
    s_error_out : np.ndarray = LavaPyType(float, float)

    v_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    # third factor input
    a_third_factor_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def spiking_activation(self):
        """Spiking activation function for Learning LIF.
        """
        return self.v > self.vth

    def calculate_third_factor_trace_y2(self, s_error_in: float) -> float: 
        """Generate's a third factor Reward traces based on 
        Graded input spikes to the Learning LIF process. 

        For SuperSpike:
        This is the error signal propogated from the input. 
        """
        # Decay constants for error
        # error_tau_rise = 200
        error_tau_decay = 10

        # Spike error at each time step
        error = s_error_in - self.s_out_buff        
        self.s_error_out = self.s_error_out + error

        # error trace updated with rise constant
        # self.s_error_out = np.exp(-1 / error_tau_rise) * self.s_error_out
        
        # Decaying error trace
        self.s_error_out = np.exp(-1 / error_tau_decay) * self.s_error_out

        return self.s_error_out
    
    def calculate_third_factor_trace_y3(self) -> float: 
        """Generate's a third factor Reward traces based on 
        Graded input spikes to the Learning LIF process. 

        For SuperSpike:
        This is calculating the surrogate gradient of the membrane potential. 
        """
        h_i = 0.1 * (self.v - self.vth)
        surrogate_v = np.power((1 + np.abs(h_i)), (-2))

        return surrogate_v

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the 
        Dense process for learning.
        """
        super().run_spk()
        
        a_third_factor_in = self.a_third_factor_in.recv()

        self.y1 = self.y1.astype(bool) | self.s_out_buff.astype(bool)
        self.y2 = self.calculate_third_factor_trace_y2(a_third_factor_in)
        self.y3 = self.calculate_third_factor_trace_y3()
    
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)

        self.v_port.send(self.v)

        if self.time_step % self.proc_params['learning_rule'].t_epoch == 0:
            self.y1 = self.y1 & False

####################################################################
# Creating a LearningDenseProbe for Measuring Trace Dynamics
####################################################################
class LearningDenseProbe(LearningDense):
    def __init__(self,
                 *,
                 weights: np.ndarray,
                 name= None,
                 num_message_bits = 0,
                 log_config = None,
                 learning_rule: LoihiLearningRule = None,
                 **kwargs) -> None:

        super().__init__(weights=weights,
                         name=name,
                         num_message_bits=num_message_bits,
                         log_config=log_config,
                         learning_rule=learning_rule,
                         **kwargs) 
        
        self.x1_port = OutPort(shape=(weights.shape[1], ))
        self.x2_port = OutPort(shape=(weights.shape[1], ))
        self.y2_port = OutPort(shape=(weights.shape[0], ))
        self.y3_port = OutPort(shape=(weights.shape[0], ))
        self.weights_port = OutPort(shape=weights.shape)
        self.tag_1_port = OutPort(shape=weights.shape)
        self.tag_2_port = OutPort(shape=weights.shape)


@implements(proc=LearningDenseProbe, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLearningDenseProbeModelFloat(PyLearningDenseModelFloat):
    """Implementation of Conn Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """
    x1_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    x2_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    y2_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    y3_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    tag_1_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    tag_2_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


    def __init__(self, proc_params):
        super().__init__(proc_params)

    def run_spk(self):
        super().run_spk()
        
        self.x1_port.send(self.x1)
        self.x2_port.send(self.x2)
        self.y2_port.send(self.y2)
        self.y3_port.send(self.y3)
        self.weights_port.send(self.weights)
        self.tag_1_port.send(self.tag_1)
        self.tag_2_port.send(self.tag_2)


####################################################################
# Custom Monitor Process and ProcessModel to Measure Trace State.
####################################################################
class MyMonitor(AbstractProcess):
    def __init__(self, shape_pre, shape_post, shape_learning_dense, buffer_size):
        super().__init__(shape_pre=shape_pre, 
                         shape_post=shape_post, 
                         shape_learning_dense=shape_learning_dense, 
                         buffer_size=buffer_size)
        shape_pre_buffer = shape_pre + (buffer_size, )
        shape_post_buffer = shape_post + (buffer_size, )
        shape_learning_dense_buffer = shape_learning_dense + (buffer_size, )
        
        self.pre_spikes_port = InPort(shape=shape_pre)
        self.pre_spikes = Var(shape=shape_pre_buffer, init=np.zeros(shape_pre_buffer))
        self.post_spikes_port = InPort(shape=shape_post)
        self.post_spikes = Var(shape=shape_post_buffer, init=np.zeros(shape_post_buffer))

        self.lif_mem_voltage_port = InPort(shape=shape_pre)
        self.lif_mem_voltage = Var(shape=shape_pre_buffer, init=np.zeros(shape_pre_buffer))

        self.learning_dense_x1_port = InPort(shape=shape_pre)
        self.learning_dense_x1 = Var(shape=shape_pre_buffer, init=np.zeros(shape_pre_buffer))
        self.learning_dense_x2_port = InPort(shape=shape_pre)
        self.learning_dense_x2 = Var(shape=shape_pre_buffer, init=np.zeros(shape_pre_buffer))
        self.learning_dense_y2_port = InPort(shape=shape_post)
        self.learning_dense_y2 = Var(shape=shape_post_buffer, init=np.zeros(shape_post_buffer))
        self.learning_dense_y3_port = InPort(shape=shape_post)
        self.learning_dense_y3 = Var(shape=shape_post_buffer, init=np.zeros(shape_post_buffer))
        self.learning_dense_weights_port = InPort(shape=shape_learning_dense)
        self.learning_dense_weights = Var(shape=shape_learning_dense_buffer, 
                                          init=np.zeros(shape_learning_dense_buffer))
        self.learning_dense_tag_1_port = InPort(shape=shape_learning_dense)
        self.learning_dense_tag_1 = Var(shape=shape_learning_dense_buffer, 
                                        init=np.zeros(shape_learning_dense_buffer))
        self.learning_dense_tag_2_port = InPort(shape=shape_learning_dense)
        self.learning_dense_tag_2 = Var(shape=shape_learning_dense_buffer, 
                                        init=np.zeros(shape_learning_dense_buffer))


@implements(proc=MyMonitor, protocol=LoihiProtocol)
@requires(CPU)
class MyMonitorPM(PyLoihiProcessModel):
    pre_spikes_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    pre_spikes: np.ndarray = LavaPyType(float, float)
    post_spikes_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    post_spikes: np.ndarray = LavaPyType(float, float)
    lif_mem_voltage_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    lif_mem_voltage: np.ndarray = LavaPyType(float, float)
        
    learning_dense_x1_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_x1: np.ndarray = LavaPyType(float, float)
    learning_dense_x2_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_x2: np.ndarray = LavaPyType(float, float)
    learning_dense_y2_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_y2: np.ndarray = LavaPyType(float, float)
    learning_dense_y3_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_y3: np.ndarray = LavaPyType(float, float)
    learning_dense_weights_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_weights: np.ndarray = LavaPyType(float, float)
    learning_dense_tag_1_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_tag_1: np.ndarray = LavaPyType(float, float)
    learning_dense_tag_2_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    learning_dense_tag_2: np.ndarray = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._buffer_size = proc_params["buffer_size"]

    def run_spk(self):
        pre_spikes = self.pre_spikes_port.recv()
        post_spikes = self.post_spikes_port.recv()
        lif_mem_voltage = self.lif_mem_voltage_port.recv()
        learning_dense_x1 = self.learning_dense_x1_port.recv()
        learning_dense_x2 = self.learning_dense_x2_port.recv()
        learning_dense_y2 = self.learning_dense_y2_port.recv()
        learning_dense_y3 = self.learning_dense_y3_port.recv()
        learning_dense_weights = self.learning_dense_weights_port.recv()
        learning_dense_tag_1 = self.learning_dense_tag_1_port.recv()
        learning_dense_tag_2 = self.learning_dense_tag_2_port.recv()
        
        self.pre_spikes[..., self.time_step % self._buffer_size] = pre_spikes
        self.post_spikes[..., self.time_step % self._buffer_size] = post_spikes
        self.lif_mem_voltage[..., self.time_step % self._buffer_size] = lif_mem_voltage
        self.learning_dense_x1[..., self.time_step % self._buffer_size] = learning_dense_x1
        self.learning_dense_x2[..., self.time_step % self._buffer_size] = learning_dense_x2
        self.learning_dense_y2[..., self.time_step % self._buffer_size] = learning_dense_y2
        self.learning_dense_y3[..., self.time_step % self._buffer_size] = learning_dense_y3
        self.learning_dense_weights[..., self.time_step % self._buffer_size] = learning_dense_weights
        self.learning_dense_tag_1[..., self.time_step % self._buffer_size] = learning_dense_tag_1
        self.learning_dense_tag_2[..., self.time_step % self._buffer_size] = learning_dense_tag_2


####################################################################
# Plotting functions for spikes and traces
####################################################################
def plot_spikes(spikes, legend, colors):
    offsets = list(range(1, len(spikes) + 1))
    
    plt.figure(figsize=(20, 3))
    
    spikes_plot = plt.eventplot(positions=spikes, 
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)
    
    plt.title("Spike arrival")
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")
    plt.yticks(ticks=offsets, labels=legend)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.9)
    plt.minorticks_on()
    plt.grid()
    
    plt.show()

def plot_spikes_shorter(spikes, legend, colors, x_ticks):
    offsets = list(range(1, len(spikes) + 1))
    
    plt.figure(figsize=(20, 3))
    
    spikes_plot = plt.eventplot(positions=spikes, 
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)
    
    plt.title("Spike arrival")
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")
    plt.xticks(ticks=x_ticks)
    plt.yticks(ticks=offsets, labels=legend)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.9)
    plt.minorticks_on()
    plt.grid()
    
    plt.show()
    
def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    
    plt.step(time, time_series, color=color)
    
    
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.9)
    plt.minorticks_on()
    plt.grid()
    
    plt.show()

def plot_time_series_combo(time, time_series, ylabel, title, figsize, colors, spikes, spikes_target=None, target=False):
    plt.figure(figsize=figsize)
    
    ax1=plt.subplot(1,1,1)
    ax1.step(time, time_series, color=colors[0])

    offsets = list(range(1, len(spikes) + 1))
    ax2=plt.subplot(111)
    if target:
        ax3=plt.subplot(111)
    for s in range(len(spikes)):
        ax2.axvspan(xmin=spikes[s], xmax=spikes[s]+0.1, color=colors[1])  
        if target:
            for u in range(len(spikes_target)):
                ax3.axvspan(xmin=spikes_target[u], xmax=spikes_target[u]+0.1, color=colors[2])  
        

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.9)
    plt.minorticks_on()

    
    plt.show()