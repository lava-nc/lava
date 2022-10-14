import numpy as np

from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.proc.dense.process import LearningDense, Dense 
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

# Learning rule coefficient
A_plus = -2
A_minus = 2

learning_rate = 1

# Trace decay constants
x1_tau = 10
y1_tau = 10

# Eligibility trace decay constant
tag_tau = "2 ^ -1" 

# error signal decay constant
y2_tau = 2 ** 32-1 
# surrogate gradient of membrane potential u, decay constant
y3_tau = 2 ** 32-1 

# Impulses
x1_impulse = 16
y1_impulse = 16

# Zero impulse value for reward. 
y2_impulse = 0
y3_impulse = 0

# Epoch length
t_epoch = 2

# String learning rule for dt : eligibility trace represented as t
dt = f"u0 * y3 * x1 - u0 * {tag_tau} * t"

# String learning rule for dw
dw = f"{learning_rate} * u0 * y2 * t"

# Create custom LearningRule
SuperSpike = LoihiLearningRule(dw=dw,
                           dt=dt,
                         x1_impulse=x1_impulse,
                         x1_tau=x1_tau,
                         y1_impulse=y1_impulse,
                         y1_tau=y1_tau,
                         y2_impulse=y2_impulse,
                         y2_tau=y2_tau,
                         y3_impulse=y3_impulse,
                         y3_tau=y3_tau,
                         t_epoch=t_epoch)


# Set this tag to "fixed_pt" or "floating_pt" to choose the corresponding models.
SELECT_TAG = "floating_pt"

# LIF parameters : Only supports floating_pt for now. 
if SELECT_TAG == "floating_pt":
    du = 1
    dv = 1

vth = 240

# Number of pre-synaptic neurons per layer
num_neurons_pre = 1
shape_lif_pre = (num_neurons_pre, )
shape_conn_pre = (num_neurons_pre, num_neurons_pre)

# Number of post-synaptic neurons per layer
num_neurons_post = 1
shape_lif_post = (num_neurons_post, )
shape_conn_post = (num_neurons_post, num_neurons_post)

# Connection parameters

# SpikePattern -> LIF connection weight : PRE-synaptic
wgt_inp_pre = np.eye(num_neurons_pre) * 250

# SpikePattern -> LIF connection weight : POST-synaptic
wgt_inp_post = np.eye(num_neurons_post) * 250

# Error SpikePattern -> LIF connection weight : ERROR
wgt_inp_reward = np.eye(num_neurons_post) * 250

# LIF -> LIF connection initial weight (learning-enabled)
wgt_plast_conn = np.full(shape_conn_post, 50)
    
# Number of simulation time steps
num_steps = 100
time = list(range(1, num_steps + 1))

# Spike times
spike_prob = 0.03

# Create random spike rasters
np.random.seed(223)
spike_raster_pre = np.zeros((num_neurons_pre, num_steps))
np.place(spike_raster_pre, np.random.rand(num_neurons_pre, num_steps) < spike_prob, 1)

spike_raster_post = np.zeros((num_neurons_post, num_steps))
np.place(spike_raster_post, np.random.rand(num_neurons_post, num_steps) < spike_prob, 1)

# Create error spike signal
error_spike_signal = np.zeros((num_neurons_post, num_steps)) 
for index in range(num_steps):
    if index == 50:
        error_spike_signal[0][index] = 1

# Create input devices
pattern_pre = SpikeIn(data=spike_raster_pre.astype(int))
pattern_post = SpikeIn(data=spike_raster_post.astype(int))

# Create error signal input device
reward_pattern_post = SpikeIn(data=error_spike_signal.astype(float))

# Create input connectivity
conn_inp_pre = Dense(weights=wgt_inp_pre)
conn_inp_post = Dense(weights=wgt_inp_post)
conn_inp_reward = Dense(weights=wgt_inp_reward)

# Create pre-synaptic neurons
lif_pre = LIF(u=0,
              v=0,
              du=du,
              dv=du,
              bias_mant=0,
              bias_exp=0,
              vth=vth,
              shape=shape_lif_pre,
              name='lif_pre')

# Create plastic connection
plast_conn = LearningDense(weights=wgt_plast_conn,
                   learning_rule=SuperSpike,
                   name='plastic_dense')

# Create post-synaptic neuron
lif_post = LearningLIF(u=0,
               v=0,
               du=du,
               dv=du,
               bias_mant=0,
               bias_exp=0,
               vth=vth,
               shape=shape_lif_post,
               name='lif_post')

# Connect network
pattern_pre.s_out.connect(conn_inp_pre.s_in)
conn_inp_pre.a_out.connect(lif_pre.a_in)

pattern_post.s_out.connect(conn_inp_post.s_in)
conn_inp_post.a_out.connect(lif_post.a_in)

# Reward ports
reward_pattern_post.s_out.connect(conn_inp_reward.s_in)
conn_inp_reward.a_out.connect(lif_post.a_third_factor_in)

lif_pre.s_out.connect(plast_conn.s_in)
plast_conn.a_out.connect(lif_post.a_in)

# Connect back-propagating action potential (BAP)
lif_post.s_out_bap.connect(plast_conn.s_in_bap)

# Connect reward trace callback (y2, y3)
lif_post.s_out_y2.connect(plast_conn.s_in_y2)
lif_post.s_out_y3.connect(plast_conn.s_in_y3)

# Create monitors
mon_pre_trace = Monitor()
mon_post_trace = Monitor()
mon_reward_trace = Monitor()
mon_pre_spikes = Monitor()
mon_post_spikes = Monitor()
mon_weight = Monitor()
mon_tag = Monitor()
mon_s_in_y2 = Monitor()
mon_s_in_y3 = Monitor()

# Connect monitors
mon_pre_trace.probe(plast_conn.x1, num_steps)
mon_post_trace.probe(plast_conn.y1, num_steps)
mon_reward_trace.probe(lif_post.s_out_y2, num_steps)
mon_pre_spikes.probe(lif_pre.s_out, num_steps)
mon_post_spikes.probe(lif_post.s_out, num_steps)
mon_weight.probe(plast_conn.weights, num_steps)
mon_tag.probe(plast_conn.tag_1, num_steps)

# Running
pattern_pre.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))

print("DONE")

# Stopping
pattern_pre.stop()