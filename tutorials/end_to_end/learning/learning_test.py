import numpy as np
import matplotlib.pyplot as plt
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.process.message_interface_enum import ActorType

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense

from procs.constant_pattern.process import ConstantPattern
from procs.poisson_rate_code_spike_gen.process import PoissonRateCodeSpikeGen
from procs.learning_dense_monitor.process import LearningDenseMonitor

from lava.proc.learning_dense.learning_rule.learning_rule import LearningRule
from lava.proc.learning_dense.process import LearningDense, SignMode


def plot_spikes(spikes, legend, colors):
    offsets = list(range(1, len(spikes) + 1))

    plt.figure(figsize=(15, 10))

    spikes_plot = plt.eventplot(positions=spikes,
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)

    plt.title("Spike arrival")
    plt.legend(spikes_plot, legend)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")
    plt.yticks(ticks=offsets, labels=offsets)

    plt.show()


def plot_time_series(time, time_series, ylabel, title):
    plt.figure(figsize=(15, 10))

    plt.step(time, time_series)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)

    plt.show()


def extract_stdp_weight_changes(time, spikes_pre, spikes_post, wgt):
    # Compute the weight changes for every weight change event
    w_diff = np.zeros(wgt.shape)
    w_diff[1:] = np.diff(wgt)

    w_diff_non_zero = np.where(w_diff != 0)
    dw = w_diff[w_diff_non_zero].tolist()

    # Find the absolute time of every weight change event
    time = np.array(time)
    t_non_zero = time[w_diff_non_zero]

    # Compute the difference between post and pre synaptic spike time for every weight change event
    spikes_pre = np.array(spikes_pre)
    spikes_post = np.array(spikes_post)
    dt = []
    for i in range(0, len(dw)):
        time_stamp = t_non_zero[i]
        t_post = (spikes_post[np.where(spikes_post <= time_stamp)])[-1]
        t_pre = (spikes_pre[np.where(spikes_pre <= time_stamp)])[-1]
        dt.append(t_post - t_pre)

    return np.array(dt), np.array(dw)


def plot_stdp(time, spikes_pre, spikes_post, wgt,
              on_pre_stdp, y1_impulse, y1_tau,
              on_post_stdp, x1_impulse, x1_tau):
    # Derive weight changes as a function of time differences
    diff_t, diff_w = extract_stdp_weight_changes(time, spikes_pre, spikes_post, wgt)

    # Derive learning rule coefficients
    on_pre_stdp = eval(on_pre_stdp.replace("^", "**"))
    a_neg = on_pre_stdp * y1_impulse
    on_post_stdp = eval(on_post_stdp.replace("^", "**"))
    a_pos = on_post_stdp * x1_impulse

    # Derive x-axis limit (absolute value)
    max_abs_dt = np.maximum(np.abs(np.max(diff_t)), np.abs(np.min(diff_t)))

    # Derive x-axis for learning window computation (negative part)
    x_neg = np.linspace(-max_abs_dt, 0, 1000)
    # Derive learning window (negative part)
    w_neg = a_neg * np.exp(x_neg / y1_tau)

    # Derive x-axis for learning window computation (positive part)
    x_pos = np.linspace(0, max_abs_dt, 1000)
    # Derive learning window (positive part)
    w_pos = a_pos * np.exp(- x_pos / x1_tau)

    plt.figure(figsize=(15, 10))

    plt.scatter(diff_t, diff_w, label="Weight changes", color="b")

    plt.plot(x_neg, w_neg, label="W-", color="r")
    plt.plot(x_pos, w_pos, label="W+", color="g")

    plt.title("STDP weight changes - Learning window")
    plt.xlabel('t_post - t_pre')
    plt.ylabel('Weight change')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    # Parameters
    SELECT_TAG = "fixed_pt"

    # Number of simulation time steps
    num_steps = 500
    time = list(range(1, num_steps + 1))

    # ConstantPattern + PoissonRateSpikeGen + LIF shape (1 neuron)
    nb_neurons = 1

    shape = (nb_neurons,)
    # Connections shape (1 -> 1 neurons)
    conn_shape = shape + shape

    # Firing rate of pre-synaptic neuron
    rate_pre = 0.04
    # Firing rate of post-synaptic neuron
    rate_post = 0.04

    # Seeds for Poisson spike generation
    seed_spike_gen_pre = 0
    seed_spike_gen_post = 1

    # LIF parameters
    if SELECT_TAG == "fixed_pt":
        du = 4095
        dv = 4095
    elif SELECT_TAG == "floating_pt":
        du = 1
        dv = 1
    vth = 240

    # PoissonRateSpikeGen -> LIF connection weight
    wgt_sg_lif = 250
    wgt_sg_lif_matrix = np.full(conn_shape, wgt_sg_lif)

    # LIF -> LIF connection initial weight (learning-enabled)
    wgt_pre_post = 50
    wgt_pre_post_matrix = np.full(conn_shape, wgt_pre_post)

    pattern_pre = ConstantPattern(shape=shape, init_value=rate_pre)
    spike_gen_pre = PoissonRateCodeSpikeGen(shape=shape,
                                            seed=seed_spike_gen_pre)
    conn_sg_pre = Dense(weights=wgt_sg_lif_matrix)
    lif_pre = LIF(u=0,
                  v=0,
                  du=du,
                  dv=du,
                  bias_mant=0,
                  bias_exp=0,
                  vth=vth,
                  shape=shape)

    pattern_post = ConstantPattern(shape=shape, init_value=rate_post)
    spike_gen_post = PoissonRateCodeSpikeGen(shape=shape,
                                             seed=seed_spike_gen_post)
    conn_sg_post = Dense(weights=wgt_sg_lif_matrix)
    lif_post = LIF(u=0,
                   v=0,
                   du=du,
                   dv=dv,
                   bias_mant=0,
                   bias_exp=0,
                   vth=vth,
                   shape=shape)

    # Learning rule coefficient
    on_pre = "0"
    on_post = "0"
    on_pre_stdp = "(-2)"
    on_post_stdp = "4"

    learning_rate = "1"

    # String learning rule for dw
    dw = "4 * y0 * x1 - 2 * x0 * y1"
    dw = f"{learning_rate} * {on_pre} * x0 + " \
         f"{learning_rate} * {on_post} * y0 + " \
         f"{learning_rate} * {on_pre_stdp} * x0 * y1 + " \
         f"{learning_rate} * {on_post_stdp} * y0 * x1"

    # Trace impulse values
    x1_impulse = 16
    y1_impulse = 16

    # Trace decay constants
    x1_tau = 10
    y1_tau = 10

    # Epoch length
    t_epoch = 2

    # Instantiating LearningRule
    learning_rule = LearningRule(dw=dw,
                                 x1_impulse=x1_impulse, x1_tau=x1_tau,
                                 y1_impulse=y1_impulse, y1_tau=y1_tau,
                                 t_epoch=t_epoch)

    # Instantiating LearningConn Process
    conn_pre_post = LearningDense(initial_weights=wgt_pre_post_matrix,
                                  learning_rule=learning_rule,
                                  sign_mode=SignMode.EXCITATORY)

    # Connecting Processes
    pattern_pre.a_out.connect(spike_gen_pre.a_in)
    spike_gen_pre.s_out.connect(conn_sg_pre.s_in)
    conn_sg_pre.a_out.connect(lif_pre.a_in)

    pattern_post.a_out.connect(spike_gen_post.a_in)
    spike_gen_post.s_out.connect(conn_sg_post.s_in)
    conn_sg_post.a_out.connect(lif_post.a_in)

    # Connecting LearningConn Process
    lif_pre.s_out.connect(conn_pre_post.s_in)
    conn_pre_post.a_out.connect(lif_post.a_in)

    # bAP (back-propagating action potentials) connection: receives spikes from post-synaptic neurons
    lif_post.s_out.connect(conn_pre_post.s_in_bap)

    # Probes
    monitor = LearningDenseMonitor(shape=conn_shape, buffer=num_steps)
    lif_pre.s_out.connect(monitor.s_pre_port)
    lif_post.s_out.connect(monitor.s_post_port)
    monitor.x1_port.connect_var(conn_pre_post.x1)
    monitor.y1_port.connect_var(conn_pre_post.y1)
    monitor.weights_port.connect_var(conn_pre_post.weights)

    # Running
    pattern_pre.run(condition=RunSteps(num_steps=num_steps),
                    run_cfg=Loihi1SimCfg(select_tag=SELECT_TAG))
    # Collecting probed data
    spikes_data_pre = monitor.s_pre.get()[0]
    spikes_pre = np.where(spikes_data_pre)[0]

    spikes_data_post = monitor.s_post.get()[0]
    spikes_post = np.where(spikes_data_post)[0]

    x1_data = monitor.x1.get()[0]
    y1_data = monitor.y1.get()[0]
    wgt_data = monitor.weights.get()[0][0]

    # Stopping
    pattern_pre.stop()

    # Plotting pre- and post- spike arrival
    plot_spikes(spikes=[spikes_post, spikes_pre],
                legend=['Post-synaptic spikes', 'Pre-synaptic spikes'],
                colors=['#370665', '#f14a16'])

    # Plotting x1 trace dynamics
    plot_time_series(time=time, time_series=x1_data, ylabel="Trace value",
                     title="x1 trace")

    # Plotting y1 trace dynamics
    plot_time_series(time=time, time_series=y1_data, ylabel="Trace value",
                     title="y1 trace")

    # Plotting weight dynamics
    plot_time_series(time=time, time_series=wgt_data, ylabel="Weight value",
                     title="Weight dynamics")

    plot_stdp(time, spikes_pre, spikes_post, wgt_data,
              on_pre_stdp, y1_impulse, y1_tau,
              on_post_stdp, x1_impulse, x1_tau)
