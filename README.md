![image](https://user-images.githubusercontent.com/68661711/135301797-400e163d-71a3-45f8-b35f-e849e8c74f0c.png)
<p align="center"><b>
  A Software Framework for Neuromorphic Computing
</b></p>

# Introduction
Lava is an open-source software framework for developing neuro-inspired applications and mapping them to neuromorphic hardware. Lava provides developers with the tools and abstractions to develop applications that fully exploit the principles of neural computation.  Constrained in this way, like the brain, Lava applications allow neuromorphic platforms to intelligently process, learn from, and respond to real-world data with great gains in energy efficiency and speed compared to conventional computer architectures.

The vision behind Lava is an open, community-developed code base that unites the full range of approaches pursued by the neuromorphic computing community. It provides a modular, composable, and extensible structure for researchers to integrate their best ideas into a growing algorithms library, while introducing new abstractions that allow others to build on those ideas without having to reinvent them.

For this purpose, Lava allows developers to define versatile _processes_ such as individual neurons, neural networks, conventionally coded programs, interfaces to peripheral devices, and bridges to other software frameworks. Lava allows collections of these processes to be encapsulated into modules and aggregated to form complex neuromorphic applications.  Communication between Lava processes uses event-based message passing, where messages can range from binary spikes to kilobyte-sized packets.

The behavior of Lava processes is defined by one or more _implementation models_, where different models may be specified for different execution platforms ("backends"), different degrees of precision, and for high-level algorithmic modeling purposes.  For example, an excitatory/inhibitory neural network process may have different implementation models for an analog neuromorphic chip compared to a digital neuromorphic chip, but the two models could share a common "E/I" process definition with each model's implementations determined by common input parameters.

Lava is platform-agnostic so that applications can be prototyped on conventional CPUs/GPUs and deployed to heterogeneous system architectures spanning both conventional processors as well as a range of neuromorphic chips such as Intel's Loihi. To compile and execute processes for different backends, Lava builds on a low-level interface called _Magma_ with a powerful compiler and runtime library. Over time, the Lava developer community may enhance Magma to target additional neuromorphic platforms beyond its initial support for Intel's Loihi chips.

The Lava framework currently supports (to be released soon):
- Channel-based message passing between asynchronous processes (the Communicating Sequential Processes paradigm)
- Hyper-granular parallelism where computation emerges as the collective result of inter-process interactions
- Heterogeneous execution platforms with both conventional and neuromorphic components
- Measurement (and cross-platform modeling) of performance and energy consumption
- Offline backprop-based training of a wide range of neuron models and network topologies
- Online real-time learning using plasticity rules constrained to access only locally available process information
- Tools for generating complex spiking neural networks such as _dynamic neural fields_ and networks that solve well-defined optimization problems
- Integration with third-party frameworks 

Future planned enhancements include support for emerging computational paradigms such as Vector Symbolic Architectures (aka Hyperdimensional Computing) and nonlinear oscillatory networks.

For maximum developer productivity, Lava blends a simple Python Interface with accelerated performance using underlying C/C++/CUDA/OpenCL code.

For more information, visit the Lava Documentation: http://lava-nc.org/

# Release plan
Intel's Neuromorphic Computing Lab (NCL) developed the initial Lava architecture as the result of an iterative (re-)design process starting from its initial Loihi Nx SDK software.  As of October, 2021, this serves as the seed of the Lava open source project, which will be released in stages beginning October 2021 as final refactoring for the new Lava software architecture is completed.
During the first two months after the initial Sept 30 launch, NCL will release the core Lava components and first algorithm libraries in regular bi-weekly releases.
		
After this first wave of releases, expect NCL releases to relax to quarterly intervals, allowing more time for significant new features and enhancements to be implemented, as well as increasing engagement with community-wide contributors.

**Initial release schedule:**
Component                        | HW support  | Features
---------------------------------| ------------| --------
Magma                            | CPU, GPU    | - The generic high-level and HW-agnostic API supports creation of processes that execute asynchronously, in parallel and communicate via messages over channels to enable algorithm and application development. <br /> - Compiler and Runtime initially only support execution or simulation on CPU and GPU platform. <br /> - A series of basic examples and tutorials explain Lava's key architectural and usage concepts
Proces library                   | CPU, GPU    | Process library initially supports basic processes to create spiking neural networks with different neuron models, connection topologies and input/output processes.
Deep Learning library            | CPU, GPU    | The Lava Deep Learning (DL) library allows for direct training of stateful and event-based spiking neural networks with backpropagation via SLAYER 2.0 as well as inference through Lava. Training and inference will initially only be supported on CPU/GPU HW.
Optimization library             | CPU, GPU    | The Lava optimization library offers a variety of constraint optimization solvers such as constraint satisfaction (CSP) or quadratic unconstraint binary optimization (QUBO) and more.
Dynamic Neural Field library     | CPU, GPU    | The Dynamic Neural Field (DNF) library allows to build neural attractor networks for working memory, decision making, basic neuronal representations, and learning.
Magma and Process library        | Loihi 1, 2  | Compiler, Runtime and the process library will be upgraded to support Loihi 1 and 2 architectures.
Profiler                         | CPU, GPU    | The Lava Profiler enable power and performance measurements on neuromorphic HW as well as the ability to simulate power and performance of neuromorphic HW on CPU/GPU platforms. Initially only CPU/GPU support will be available.
DL, DNF and Optimization library | Loihi 1, 2 | All algorithm libraries will be upgraded to support and be properly tested on neuromorphic HW.


# Lava organization
Processes are the fundamental building block in the Lava architecture from which all algorithms and applications are built. Processes are stateful objects with internal variables, input and output ports for message-based communication via channels and multiple behavioral models. This architecture is inspired from the Communicating Sequential Process (CSP) paradigm for asynchronous, parallel systems that interact via message passing. Lava processes implementing the CSP API can be compiled and executed via a cross-platform compiler and runtime that support execution on neuromorphic and conventional von-Neumann HW. Together, these components form the low-level Magma layer of Lava.
			
At a higher level, the process library contains a growing set of generic processes that implement various kinds of neuron models, neural network connection topologies, IO processes, etc. These execute on either CPU, GPU or neuromorphic HW such as Intel's Loihi architecture. 
			
Various algorithm and application libraries build on these these generic processes to create specialized processes and provide tools to train or configure processes for more advanced applications. A deep learning library, constrained optimization library, and dynamic neural field library are among the first to be released in Lava, with more libraries to come in future releases.
			
Lava is open to modification and extension to third-party libraries like Nengo, ROS, YARP and others. Additional utilities also allow users to profile power and performance of workloads, visualize complex networks, or help with the float to fixed point conversions required for many low-precision devices such as neuromorphic HW.

![image](https://user-images.githubusercontent.com/68661711/135412508-4a93e20a-8b64-4723-a69b-de8f4b5902f7.png)

All of Lava's core APIs and higher-level components are released, by default, with permissive BSD 3 licenses in order to encourage the broadest possible community contribution.  Lower-level Magma components needed for mapping processes to neuromorphic backends are generally released with more restrictive LGPL-2.1 licensing to discourage commercial proprietary forks of these technologies.  The specific components of Magma needed to compile processes specifically to Intel Loihi chips remains proprietary to Intel and is not provided through this GitHub site (see below).  Similar Magma-layer code for other future commercial neuromorphic platforms likely will also remain proprietary.

# Getting started
## Install instructions
### Installing or cloning Lava
New Lava releases will be published via GitHub releases and can be installed after downloading.

```console

   pip install lava-0.0.1.tar.gz
   pip install lava-lib-0.0.1.tar.gz
```

If you would like to contribute to the source code or work with the source directly, you can also clone the repository.

```console

   git clone git@github.com:lava-nc/lava.git
   pip install -e lava/lava
   
   git clone git@github.com:lava-nc/lava-lib.git
   # [Optional]
   pip install -e lava-lib/dnf
   pip install -e lava-lib/dl
   pip install -e lava-lib/optimization
```

This will allow you to run Lava on your own local CPU or GPU.

### Running Lava on Intel Loihi

Intel's neuromorphic Loihi 1 or 2 research systems are currently not available commercially. Developers interested in using Lava with Loihi systems, need to join the Intel Neuromorphic Research Community (INRC). Once a member of the INRC, developers will gain access to cloud-hosted Loihi systems or are able to obtain physical Loihi systems on a loan basis. In addition, Intel will provide further proprietary components of the magma library which enable compiling processes for Loihi systems that need to be installed into the same _Lava_ namespace as in this example:

```console

   pip install /nfs/ncl/releases/lava/0.0.1/lava-nc-0.0.1.tar.gz
   pip install /nfs/ncl/releases/lava/0.0.1/lava-nc-lib-0.0.1.tar.gz
```

Please email inrc_interest@intel.com to request a research proposal template to apply for INRC membership.


## Coding example
### Building a simple feed-forward network
```python
# Instantiate Lava processes to build network
import numpy as np
from lava.proc.io import SpikeInput, SpikeOutput
from lava.proc import Dense, LIF

si = SpikeInput(path='source_data_path', shape=(28, 28))
dense = Dense(shape=(10, 784),
              weights=np.random.random((10, 784)))
lif = LIF(shape=(10,), vth=10)
so = SpikeOutput(path='result_data_path', shape=(10,))

# Connect processes via their directional input and output ports
si.out_ports.s_out.reshape(784, 1).connect(dense.in_ports.s_in)
dense.out_ports.a_out.connect(lif.in_ports.a_in)
lif.out_ports.s_out.connect(so.in_ports.s_in)

# Execute processes for fixed number of steps on Loihi 2 (by running any of them)
from lava.magma import run_configs as rcfg
from lava.magma import run_conditions as rcnd
lif.run(run_cfg=rcfg.Loihi2HwCfg(),
        condition=rcnd.RunSteps(1000, blocking=True))
```

### Creating a custom Lava process
A process has input and output ports to interact with other processes, internal variables may have different behavioral implementations in different programming languages or for different HW platforms.
```python
from lava.magma import AbstractProcess, InPort, Var, OutPort

class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire neural process with activation input and spike
    output ports a_in and s_out.

    Realizes the following abstract behavior:
    u[t] = u[t-1] * (1-du) + a_in
    v[t] = v[t-1] * (1-dv) + u[t] + b
    s_out = v[t] > vth
    v[t] = v[t] - s_out*vth
    """
    def __init__(self, **kwargs):
        super(AbstractProcess, self).__init__(kwargs)
        shape = kwargs.pop("shape", (1,))
        # Declare input and output ports
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        # Declare internal variables
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.decay_u = Var(shape=(1,), init=kwargs.pop('du', 1))
        self.decay_v = Var(shape=(1,), init=kwargs.pop('dv', 0))
        self.b = Var(shape=shape, init=kwargs.pop('b', 0))
        self.vth = Var(shape=(1,), init=kwargs.pop('vth', 1))
```

### Creating process models
Process models are used to provide different behavioral models of a process. This Python model implements the LIF process, the Loihi synchronization protocol and requires a CPU compute resource to run.
```python
import numpy as np
from lava import magma as mg
from lava.magma.resources import CPU
from lava.magma.sync_protocol import LoihiProtocol, DONE
from lava.proc import LIF
from lava.magma.pymodel import AbstractPyProcessModel, LavaType
from lava.magma.pymodel import InPortVecDense as InPort
from lava.magma.pymodel import OutPortVecDense as OutPort

@mg.implements(proc=LIF, protocol=LoihiProtocol)
@mg.requires(CPU)
class PyLifModel(AbstractPyProcessModel):
    # Declare port implementation
    a_in: InPort =     LavaType(InPort, np.int16, precision=16)
    s_out: OutPort =   LavaType(OutPort, bool, precision=1)
    # Declare variable implementation
    u: np.ndarray =    LavaType(np.ndarray, np.int32, precision=24)
    v: np.ndarray =    LavaType(np.ndarray, np.int32, precision=24)
    b: np.ndarray =    LavaType(np.ndarray, np.int16, precision=12)
    du: int =          LavaType(int, np.uint16, precision=12)
    dv: int =          LavaType(int, np.uint16, precision=12)
    vth: int =         LavaType(int, int, precision=8)

    def run_spk(self):
        """Executed during spiking phase of synchronization protocol."""
        # Decay current
        self.u[:] = self.u * (1 - self.du)
        # Receive input activation via channel and accumulate
        activation = self.a_in.recv()
        self.u[:] += activation
        self.v[:] = self.v * (1 - self.dv) + self.u + self.b
        # Generate output spikes and send to receiver
        spikes = self.v > self.vth
        self.v[spikes] -= self.vth
        if np.any(spikes):
            self.s_out.send(spikes)
```

In contrast this process model also implements the LIF process but by structurally allocating neural network resources on a virtual Loihi 1 neuro core.
```python
from lava import magma as mg
from lava.magma.resources import Loihi1NeuroCore
from lava.proc import LIF
from lava.magma.ncmodel import AbstractNcProcessModel, LavaType, InPort, OutPort, Var

@mg.implements(proc=LIF)
@mg.requires(Loihi1NeuroCore)
class NcProcessModel(AbstractNcProcessModel):
    # Declare port implementation
    a_in: InPort =   LavaType(InPort, precision=16)
    s_out: OutPort = LavaType(OutPort, precision=1)
    # Declare variable implementation
    u: Var =         LavaType(Var, precision=24)
    v: Var =         LavaType(Var, precision=24)
    b: Var =         LavaType(Var, precision=12)
    du: Var =        LavaType(Var, precision=12)
    dv: Var =        LavaType(Var, precision=12)
    vth: Var =       LavaType(Var, precision=8)

    def allocate(self, net: mg.Net):
        """Allocates neural resources in 'virtual' neuro core."""
        num_neurons = self.in_args['shape'][0]
        # Allocate output axons
        out_ax = net.out_ax.alloc(size=num_neurons)
        net.connect(self.s_out, out_ax)
        # Allocate compartments
        cx_cfg = net.cx_cfg.alloc(size=1,
                                  du=self.du,
                                  dv=self.dv,
                                  vth=self.vth)
        cx = net.cx.alloc(size=num_neurons,
                          u=self.u,
                          v=self.v,
                          b_mant=self.b,
                          cfg=cx_cfg)
        cx.connect(out_ax)
        # Allocate dendritic accumulators
        da = net.da.alloc(size=num_neurons)
        da.connect(cx)
        net.connect(self.a_in, da)
```

# Stay in touch
To receive regular updates on the latest developments and releases of the Lava Software Framework please [subscribe to our newsletter](http://eepurl.com/hJCyhb).


