from lava.proc.rf.models import RF
import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer as RingBufferSend
from lava.proc.io.sink import RingBuffer as RingBufferReceive, PyReceiveModelFloat
from lava.proc.rf_iz.process import RF_IZ
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
def process_spikes(time_range: np.array, spikes: np.array):
    spike_mask = np.argwhere(spikes)
    toas = time_range[spike_mask]
    spike_amps = spikes[spike_mask]
    return list(zip(toas, spike_amps))

def plot_spike_graph(time_range: np.array, signal: list, spikes: np.array, real_comp: np.array, complex_threshold: int, title:str, save_name:str):
    plt.figure()
    spike_info = process_spikes(time_range, spikes)

    plt.plot(time_range, [complex_threshold] * len(time_range), label ="threshold", color = "black")
    for toa, amp in spike_info:
        plt.vlines(toa,complex_threshold, complex_threshold+amp, color = "black", linestyles="--", label = "spike")

    plt.plot(time_range, real_comp.flatten(), label = "Voltage")

    plt.plot(time_range, signal - 2, color = "orange", label = "input pulse")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.savefig(save_name)
    plt.show()

from sympy.solvers import solve
from sympy import Symbol
from sympy import solve_univariate_inequality, Interval
if __name__ == "__main__":
    vth = 1.1
    neuron1_period = 10
    neuron2_period = 9

    data_period= 10 # every ten timesteps complete and oscillation

    num_steps = 300

    alpha = Symbol('a')
    dv1 =  1 - np.power(vth - 1, 1/neuron1_period) - 1e-10
    dv2 =  1 - np.power(vth - 1, 1/neuron2_period) - 1e-10

    # create lava neurons
    neuron1_real = LIF(shape=(1,),  du=1, dv=dv1,  vth = vth) # create a single neuron
    neuron2_real = LIF(shape=(1,),  du=1, dv=dv2,  vth = vth) # create a single neuron
    
    neuron2_to_neuron1 = Dense(weights= np.array([[-3]]))
    neuron2_real.s_out.connect(neuron2_to_neuron1.s_in)
    neuron2_to_neuron1.a_out.connect(neuron1_real.a_in)


    incoming_spike_data = np.zeros((1, num_steps))  
    incoming_spike_data[:, [20, 30]] = 1
    incoming_spike_data[:, np.arange(20, num_steps, step = data_period)] = 1
    incoming_data_process = RingBufferSend(data=incoming_spike_data)
    incoming_data_process.out_ports.s_out.connect(neuron1_real.a_in) # connect incoming spikes to neuron
    incoming_data_process.out_ports.s_out.connect(neuron2_real.a_in)

    # Create a recepticle for the neurons output spikes
    outgoing_data_process = RingBufferReceive(shape=(1,), buffer =num_steps)
    neuron1_real.s_out.connect(outgoing_data_process.a_in)

    # monitor the real component of the neuron
    monitor = Monitor()
    monitor.probe(target=neuron1_real.v, num_steps=num_steps) # create a probe to observe oscillations

    run_config = Loihi1SimCfg()
    # run our simulation for n_steps
    neuron1_real.run(condition=RunSteps(num_steps=num_steps, blocking=True), run_cfg=run_config)

    probe_data_volt = monitor.get_data()[neuron1_real.name]["v"]
    probe_data_spike = outgoing_data_process.data.get()
    neuron1_real.stop()  # clears data in probe with stop
    

    plot_spike_graph(np.arange(num_steps), incoming_spike_data.flatten(), probe_data_spike.flatten(), probe_data_volt.flatten(), vth, "LIF period %d Neuron with period %d input" % (neuron1_period, data_period), "rf_lava.png")        