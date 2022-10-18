from torch import _fused_moving_avg_obs_fq_helper
from lava.proc.rf_iz.models import RF_IZ
from lava.proc.rf.models import RF
import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer as RingBufferSend
from lava.proc.io.sink import RingBuffer as RingBufferReceive, \
                                            PyReceiveModelFloat

# from lava.proc.rf_iz.process import RF_IZ


def process_spikes(time_range: np.array, spikes: np.array):
    spike_mask = np.argwhere(spikes)
    toas = time_range[spike_mask]
    spike_amps = spikes[spike_mask]
    return list(zip(toas, spike_amps))


def plot_spike_graph(time_range: np.array, signal: list, spikes: np.array, 
                     real_comp: np.array, complex_threshold: int,
                     title: str, save_name: str):
    plt.figure()
    spike_info = process_spikes(time_range, spikes)

    plt.plot(time_range, [complex_threshold] * len(time_range),
             label="threshold", color="black")
    for toa, amp in spike_info:
        plt.vlines(toa, complex_threshold, complex_threshold+amp,
                   color="black", linestyles="--", label="spike")

    plt.plot(time_range, real_comp.flatten(), label="Re (z(t))")

    plt.plot(time_range, signal - 2, color="orange", label="input pulse")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":

    n_neurons = 1  # 1 neuron
    period = 10  # every ten timesteps complete and oscillation
    alpha = .07
    fixed_point = True

    if fixed_point:
        state_exp = 6
        decay_bits = 12
    else:
        decay_bits = 0
        state_exp = 0

    vth = 1.1
    num_steps = 100

    # create lava neuron
    neuron = RF(shape=(n_neurons,), period=period, alpha=alpha,
                vth=vth, state_exp=state_exp, decay_bits=decay_bits)

    # create spiking data that just spike at regular intervals
    incoming_spike_data = np.zeros((n_neurons, num_steps))
    incoming_spike_data[:, [0]] = 1
    incoming_data_process = RingBufferSend(data=incoming_spike_data)
    incoming_data_process.out_ports.s_out.connect(neuron.a_real_in)

    # Create a receptacle for the neurons output spikes
    outgoing_data_process = RingBufferReceive(shape=(n_neurons,),
                                              buffer=num_steps)
    neuron.s_out.connect(outgoing_data_process.a_in)

    # monitor the real component of the neuron
    monitor = Monitor()
    monitor.probe(target=neuron.real, num_steps=num_steps)  # observe internals
    if fixed_point:
        class CustomRunConfig(Loihi1SimCfg):
            def select(self, proc, proc_models):
                if isinstance(proc, RingBufferReceive):
                    return PyReceiveModelFloat
                else:
                    return super().select(proc, proc_models)

        run_config = CustomRunConfig(select_tag='fixed_pt')
    else:
        run_config = Loihi1SimCfg()

    # run our simulation for n_steps
    neuron.run(condition=RunSteps(num_steps=num_steps, blocking=True),
               run_cfg=run_config)

    probe_data_real = monitor.get_data()[neuron.name]["real"]
    probe_data_spike = outgoing_data_process.data.get()
    neuron.stop()  # clears data in probe with stop
    
    assert n_neurons == 1
    if fixed_point:
        plot_spike_graph(np.arange(num_steps),
                         incoming_spike_data.flatten() > 0,
                         probe_data_spike.flatten() > 0,
                         probe_data_real.flatten()/(1 << 6), vth,
                         "Rf Neuron in Lava Fixed point", "rf_lava.png")
    else:
        plot_spike_graph(np.arange(num_steps),
                         incoming_spike_data.flatten() > 0,
                         probe_data_spike.flatten() > 0,
                         probe_data_real.flatten(), vth,
                         "Rf Neuron in Lava float point", "rf_lava.png")