# Implementation of SuperSpike Learning rule in Lava

This folder contains the implementation of the SuperSpike learning rule. The initial paper implements a double exponential filter applied to the error trace, pre and post-synaptic trace which is implemented with the help of the difference of exponential formula and also the elgibility trace. 

Refer to the paper for more details : Zenke, F., and Ganguli, S. (2018). SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation 30, 1514–1541.

URL: https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01086

The publically available implementation might provide some of the values for the simulation parameters : [SuperSpike](https://github.com/fzenke/pub2018superspike)

Learning rule: 

> dt represents the eligibility trace with the pre and post-synaptic traces along with the surrogate gradient trace (y3)

> dd decays the eligibility trace (implements the double exponential filter)

> dw is ffunction of d and the error trace (y2)

### POINTS TO REMEMBER:

utils_SuperSpike.py: 

> Implements a floating point version of the SuperSpikeLIF neuron.

  > error_tau_rise and error_tau_decay should be symmetric to the 
      eligibility_trace_decay_tau and eligibility_trace_rise_tau.

  > y1 implements the filtering of post-synaptic trace (replicating a ucoded neuron)

> LearningDenseProbe Process and ProcessModels are created to monitor "Var". It replaces "Var's" as InPorts or OutPorts for measuring. This was to circumvent an issue with Var monitor slowing down the simulation significantly. A simple LearningDense is replaced with LearningDenseProb in simulation. 
Note :  This can be removed later - if the Monitor problem is fixed! 

> MyMonitor Process and ProcModel connects to the Probe class to record state of simulation. [LearningDenseProbe -> MyMonitor]

tutorial_SuperSpike_DoubleExponential

> The network learns with exactly two pre-synaptic spike and one target spike. This simple prototype has one pre-synaptic neuron and one post-synaptic neuron. 

> There is one timestep difference in the final output spikes due to the monitoring happening in the post management phase. However we can ignore it and consider it to be spiking at the target spike time. 