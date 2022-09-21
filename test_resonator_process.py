from lava.proc.rf.models import RF
import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer as RingBufferSend
from lava.proc.io.sink import RingBuffer as RingBufferReceive, PyReceiveModelFloat

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

    plt.plot(time_range, real_comp.flatten(), label = "Re (z(t))")

    plt.plot(time_range, signal - 2, color = "orange", label = "input pulse")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":
    n_neurons = 1 # 1 neuron
    period = 10 # every ten timesteps complete and oscillation
    alpha = .07
    frequency = 1/period
    radians_per_second = frequency * np.pi * 2
    fixed_point = True
    sin_decay = (1 - alpha) * np.sin(radians_per_second)
    cos_decay = (1 - alpha) * np.cos(radians_per_second)

    p_scale = 1 << 12
    if fixed_point:
        sin_decay = int(sin_decay * (1 << 12)) - 1
        cos_decay = int(cos_decay * (1 << 12)) - 1

    vth = 1
    num_steps = 100

    # create lava neuron
    neuron = RF(shape=(n_neurons,), sin_decay = sin_decay, cos_decay = cos_decay, vth = vth) # create a single neuron

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
    monitor.probe(target=neuron.real, num_steps=num_steps) # create a probe to observe oscillations
    if fixed_point:
        class CustomRunConfig(Loihi1SimCfg):
            def select(self, proc, proc_models):
                # customize run config to always use float model for io.sink.RingBuffer
                if isinstance(proc, RingBufferReceive):
                    return PyReceiveModelFloat
                else:
                    return super().select(proc, proc_models)

        run_config = CustomRunConfig(select_tag='fixed_pt')
    else:
        run_config = Loihi1SimCfg()

    # run our simulation for n_steps
    neuron.run(condition=RunSteps(num_steps=num_steps, blocking=True), run_cfg=run_config)

    probe_data_real = monitor.get_data()[neuron.name]["real"]
    probe_data_spike = outgoing_data_process.data.get()
    neuron.stop()  # clears data in probe with stop
    
    assert n_neurons == 1, "Rest of Logic will not work if there are more than 1 neurons"
    if fixed_point:
        plot_spike_graph(np.arange(num_steps), incoming_spike_data.flatten() > 0, probe_data_spike.flatten() > 0, probe_data_real.flatten()/(1 << 6), vth, "Rf Neuron in Lava Fixed point", "rf_lava.png")
    else:
        plot_spike_graph(np.arange(num_steps), incoming_spike_data.flatten(), probe_data_spike.flatten(), probe_data_real.flatten(), vth, "Rf Neuron in Lava float point", "rf_lava.png")        