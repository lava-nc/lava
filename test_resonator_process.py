from lava.proc.rf.models import RF
from lava.proc.rf_iz.models import RF_IZ
import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer as RingBufferSend
from lava.proc.io.sink import RingBuffer as RingBufferReceive

if __name__ == "__main__":
    n_neurons = 1 # 1 neuron
    period = 10 # every ten timesteps complete and oscillation
    alpha = .07
    frequency = 1/period
    radians_per_second = frequency * np.pi * 2
    sin_decay = (1 - alpha) * np.sin(radians_per_second)
    cos_decay = (1 - alpha) * np.cos(radians_per_second)

    vth = 1
    num_steps = 100

    # create lava neuron
    neuron = RF_IZ(shape=(n_neurons,), sin_decay = sin_decay, cos_decay = cos_decay, vth = vth) # create a single neuron

    # create spiking data that just spikes at timestep 1
    incoming_spike_data = np.zeros((n_neurons, num_steps))  
    incoming_spike_data[:, [10, 20, 30, 40]] = 1  # neuron recieves input spikes at timestep 10, 20, 30, and 40
    incoming_data_process = RingBufferSend(data=incoming_spike_data)
    incoming_data_process.out_ports.s_out.connect(neuron.a_real_in) # connect incoming spikes to neuron

    # Create a recepticle for the neurons output spikes
    outgoing_data_process = RingBufferReceive(shape=(n_neurons,), buffer =num_steps)
    neuron.s_out.connect(outgoing_data_process.a_in)

    # monitor the real component of the neuron
    monitor = Monitor()
    monitor.probe(target=neuron.real, num_steps=100) # create a probe to observe oscillations

    # run our simulation for n_steps
    neuron.run(condition=RunSteps(num_steps=100, blocking=True), run_cfg=Loihi1SimCfg())
    probe_data_real = monitor.get_data()[neuron.name]["real"]
    probe_data_spike = outgoing_data_process.data.get()
    neuron.stop()  # clears data in probe with stop
    
    assert n_neurons == 1, "Rest of Logic will not work if there are more than 1 neurons"

    plt.plot(np.arange(num_steps), probe_data_real, label = "real")
    plt.title("sub-threshold dynamics of rf neuron")
    plt.xlabel("Timestep")
    plt.ylabel("Re(z)")
    plt.show()