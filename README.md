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
Process library                   | CPU, GPU    | Process library initially supports basic processes to create spiking neural networks with different neuron models, connection topologies and input/output processes.
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

## Getting Started

### Cloning Lava and Running from Source

We highly recommend cloning the repository and using poetry to setup lava.
 You will need to install poetry.

Open a **python 3** terminal and run based on the OS you are on:

#### [Linux/MacOS]
```bash
cd $HOME
pip install -U pip
pip install "poetry>=1.1.13"
git clone git@github.com:lava-nc/lava.git
cd lava
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
pytest

## When running tests if you see 'OSError: [Errno 24] Too many open files'
## consider setting ulimit using `ulimit -n 4096`
## See FAQ for more info: https://github.com/lava-nc/lava/wiki/Frequently-Asked-Questions-(FAQ)#install
```
Note that you should install the core Lava repository (lava) before installing other Lava libraries such as lava-optimization or lava-dl.

#### [Windows]
```powershell
# Commands using PowerShell
cd $HOME
git clone git@github.com:lava-nc/lava.git
cd lava
python3 -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install "poetry>=1.1.13"
poetry config virtualenvs.in-project true
poetry install
pytest
```
You should expect the following output after running the unit tests:
```
$ pytest
============================================== test session starts ==============================================
platform linux -- Python 3.8.10, pytest-7.0.1, pluggy-1.0.0
rootdir: /home/user/lava, configfile: pyproject.toml, testpaths: tests
plugins: cov-3.0.0
collected 205 items

tests/lava/magma/compiler/test_channel_builder.py .                                                       [  0%]
tests/lava/magma/compiler/test_compiler.py ........................                                       [ 12%]
tests/lava/magma/compiler/test_node.py ..                                                                 [ 13%]
tests/lava/magma/compiler/builder/test_channel_builder.py .                                               [ 13%]

...... pytest output ...

tests/lava/proc/sdn/test_models.py ........                                                               [ 98%]
tests/lava/proc/sdn/test_process.py ...                                                                   [100%]
=============================================== warnings summary ================================================

...... pytest output ...

src/lava/proc/lif/process.py                                                           38      0   100%
src/lava/proc/monitor/models.py                                                        27      0   100%
src/lava/proc/monitor/process.py                                                       79      0   100%
src/lava/proc/sdn/models.py                                                           159      9    94%   199-202, 225-231
src/lava/proc/sdn/process.py                                                           59      0   100%
-----------------------------------------------------------------------------------------------------------------TOTAL
                                                                                     4048    453    89%

Required test coverage of 85.0% reached. Total coverage: 88.81%
============================ 199 passed, 6 skipped, 2 warnings in 118.17s (0:01:58) =============================

```

### [Alternative] Installing Lava from Binaries

If you only need the lava package in your python environment, we will publish
Lava releases via
[GitHub Releases](https://github.com/lava-nc/lava/releases). Please download
the package and install it.

Open a python terminal and run:

#### [Windows/MacOS/Linux]
```bash
python -m venv .venv
source .venv/bin/activate ## Or Windows: .venv\Scripts\activate
pip install -U pip
pip install lava-nc-0.3.0.tar.gz
```

### Linting, Testing, Documentation and Packaging

```bash
# Install poetry
pip install "poetry>=1.1.13"
poetry config virtualenvs.in-project true
poetry install
poetry shell

# Run linting
flakeheaven lint src/lava tests

# Run unit tests
pytest

# Create distribution
poetry build
#### Find builds at dist/

# Run Secuity Linting
bandit -r src/lava/.

#### If security linting fails run bandit directly
#### and format failures
bandit -r src/lava/. --format custom --msg-template '{abspath}:{line}: {test_id}[bandit]: {severity}: {msg}'
```

## Running Lava on Intel Loihi

Intel's neuromorphic Loihi 1 or 2 research systems are currently not available
commercially. Developers interested in using Lava with Loihi systems, need to
join the Intel Neuromorphic Research Community (INRC). Once a member of the
INRC, developers will gain access to cloud-hosted Loihi systems or are able
to obtain physical Loihi systems on a loan basis. In addition, Intel will
provide further proprietary components of the magma library which enable
compiling processes for Loihi systems that need to be installed into the
same Lava namespace.

Login to Intel External vLab Machines. Instructions were sent to you during
your INRC Account Setup. If you do not know/remember the login
instructions, email: nrc_support@intel-research.net

- Login to INRC VM with your credentials
- Follow the instructions to Install or Clone Lava
- cd /nfs/ncl/releases/lava/0.3.0
- pip install lava-nc-0.3.0.tar.gz

## Coding example
### Building a simple feed-forward network
```python
# Instantiate Lava processes to build network
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF

lif1 = LIF()
dense = Dense()
lif2 = LIF()

# Connect processes via their directional input and output ports
lif1.out_ports.s_out.connect(dense.in_ports.s_in)
dense.out_ports.a_out.connect(lif2.in_ports.a_in)

# Execute process lif1 and all processes connected to it for fixed number of steps
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

lif1.run(condition=RunSteps(num_steps=10),
         run_cfg=Loihi1SimCfg()
)
lif1.stop()

```


### Creating a custom Lava process
A process has input and output ports to interact with other processes, internal variables may have different behavioral implementations in different programming languages or for different HW platforms.
```python
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire neural process with activation input and spike
    output ports a_in and s_out.

    Realizes the following abstract behavior:
    u[t] = u[t-1] * (1-du) + a_in
    v[t] = v[t-1] * (1-dv) + u[t] + bias
    s_out = v[t] > vth
    v[t] = v[t] - s_out*vth
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=kwargs.pop("du", 0))
        self.dv = Var(shape=(1,), init=kwargs.pop("dv", 0))
        self.bias = Var(shape=shape, init=kwargs.pop("b", 0))
        self.vth = Var(shape=(1,), init=kwargs.pop("vth", 10))
        
```

### Creating process models
Process models are used to provide different behavioral models of a process. This Python model implements the LIF process, the Loihi synchronization protocol and requires a CPU compute resource to run.
```python
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.process import LIF


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
class PyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=12)
    du: int = LavaPyType(int, np.uint16, precision=12)
    dv: int = LavaPyType(int, np.uint16, precision=12)
    vth: int = LavaPyType(int, int, precision=8)

    def run_spk(self):
        self.u[:] = self.u * ((2 ** 12 - self.du) // 2 ** 12)
        a_in_data = self.a_in.recv()
        self.u[:] += a_in_data
        self.v[:] = self.v * \
            ((2 ** 12 - self.dv) // 2 ** 12) + self.u + self.bias
        s_out = self.v > self.vth
        self.v[s_out] = 0  # Reset voltage to 0. This is Loihi-1 compatible.
        self.s_out.send(s_out)
        
```

In contrast this process model also implements the LIF process but by structurally allocating neural network resources on a virtual Loihi 1 neuro core.
```python
from lava.proc.lif.process import LIF
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import Loihi1NeuroCore
from lava.magma.core.model.nc.model import NcLoihiProcessModel
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.type import LavaNcType, NcVar

@implements(proc=LIF) #Note that the NcLoihiProcessModel class implies the useage of the Loihi SyncProtcol
@requires(Loihi1NeuroCore)
class NcLifModel(NcLoihiProcessModel):
   # Declare port implementation
   a_in: InPort =   LavaNcType(NcInPort, precision=16)
   s_out: OutPort = LavaNcType(NcOutPort, precision=1)
   # Declare variable implementation
   u: NcVar =         LavaNcType(NcVar, precision=24)
   v: NcVar =         LavaNcType(NcVar, precision=24)
   b: NcVar =         LavaNcType(NcVar, precision=12)
   du: NcVar =        LavaNcType(NcVar, precision=12)
   dv: NcVar =        LavaNcType(NcVar, precision=12)
   vth: NcVar =       LavaNcType(NcVar, precision=8)

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
