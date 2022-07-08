![image](https://user-images.githubusercontent.com/68661711/135301797-400e163d-71a3-45f8-b35f-e849e8c74f0c.png)
<p align="center"><b>
  A Software Framework for Neuromorphic Computing
</b></p>

##
>**Detailed Lava documentation:** http://lava-nc.org/

# Overview: What is Lava?

Lava is an open-source software framework for developing neuro-inspired
applications and mapping them to neuromorphic hardware. Lava provides developers
with the tools and abstractions to develop applications that fully exploit the
principles of neural computation. Constrained in this way, like the brain, Lava
applications allow neuromorphic platforms to intelligently process, learn from,
and respond to real-world data with great gains in energy efficiency and speed
compared to conventional computer architectures.

Lava is platform-agnostic so that applications can be prototyped on conventional
CPUs/GPUs and deployed to heterogeneous system architectures spanning both
conventional processors as well as a range of neuromorphic chips such as Intel's
Loihi. To compile and execute processes for different backends, Lava builds on a
low-level interface called _Magma_ with a powerful compiler and runtime library.
Over time, the Lava developer community may enhance Magma to target additional
neuromorphic platforms beyond its initial support for Intel's Loihi chips.

The Lava framework supports:

- Channel-based message passing between asynchronous processes (the
  Communicating Sequential Processes paradigm)
- Hyper-granular parallelism where computation emerges as the collective result
  of inter-process interactions
- Heterogeneous execution platforms with both conventional and neuromorphic
  components
- Measurement (and cross-platform modeling) of performance and energy
  consumption
- Offline backprop-based training of a wide range of neuron models and network
  topologies
- Online real-time learning using plasticity rules constrained to access only
  locally available process information
- Tools for generating complex spiking neural networks such as _dynamic neural
  fields_ and networks that solve well-defined optimization problems
- Integration with third-party frameworks

Lava blends a simple Python Interface with accelerated performance using 
underlying C/C++/CUDA/OpenCL code, which maximizes developer productivity. 

For detailed documentation on how Lava is organized, see http://lava-nc.org.

# Layers of Lava

## Magma
Magma is the bottom-most layer of the Lava framework, which consists of a 
cross-platform compiler and runtime that support execution on neuromorphic 
and conventional von-Neumann hardware platforms.

## Lava Processes
_Lava Processes_ are stateful objects with internal variables, input and output 
ports for message-based communication via channels and multiple behavioral 
models.

## Lava Process Libraries
At a higher level, the process library contains a growing set of generic
Lava processes that implement various kinds of neuron models, neural network
connection topologies, IO processes, etc. These execute on either CPU, GPU or
neuromorphic HW such as Intel's Loihi architecture.

Various algorithm and application libraries build on these these generic
processes to create specialized processes and provide tools to train or
configure processes for more advanced applications. A deep learning library,
constrained optimization library, and dynamic neural field library are among the
first to be released in Lava, with more libraries to come in future releases.

![image](https://user-images.githubusercontent.com/68661711/135412508-4a93e20a-8b64-4723-a69b-de8f4b5902f7.png)

# Getting Started

## Cloning Lava and Running from Source

We highly recommend cloning the repository and using poetry to setup lava.
You will need to install poetry.

Open a **python 3** terminal and run based on the OS you are on:

### [Linux/MacOS]

```bash
cd $HOME
pip install -U pip
pip install "poetry>=1.1.13"
git clone git@github.com:lava-nc/lava.git
cd lava
git checkout v0.4.0
./utils/githook/install-hook.sh
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
pytest

## When running tests if you see 'OSError: [Errno 24] Too many open files'
## consider setting ulimit using `ulimit -n 4096`
## See FAQ for more info: https://github.com/lava-nc/lava/wiki/Frequently-Asked-Questions-(FAQ)#install
```

Note that you should install the core Lava repository (lava) before installing
other Lava libraries such as lava-optimization or lava-dl.

### [Windows]

```powershell
# Commands using PowerShell
cd $HOME
git clone git@github.com:lava-nc/lava.git
cd lava
git checkout v0.4.0
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

## [Alternative] Installing Lava via Conda

If you use the conda package manager, you can simply install the lava package
via:

```bash
conda install lava -c conda-forge
```

## [Alternative] Installing Lava from Binaries

If you only need the lava package in your python environment, we will publish
Lava releases via
[GitHub Releases](https://github.com/lava-nc/lava/releases). Please download
the package and install it.

Open a python terminal and run:

### [Windows/MacOS/Linux]

```bash
python -m venv .venv
source .venv/bin/activate ## Or Windows: .venv\Scripts\activate
pip install -U pip
pip install lava-nc-0.3.0.tar.gz
```

## Linting, Testing, Documentation and Packaging

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
##
> Refer to the tutorials directory for in-depth as well as end-to-end 
> tutorials on how to write Lava Processes, connect them, and execute the code.

# Running Lava on Intel Loihi

> Lava code that enables execution on Intel Loihi is available only to the 
> engaged INRC members.

Developers interested in using Lava with Loihi systems, need to
join the Intel Neuromorphic Research Community (INRC), as the Loihi 1 or 2 
research systems are currently not available commercially. Once a member of the
INRC, developers will gain access to cloud-hosted Loihi systems or are able
to obtain physical Loihi systems on a loan basis. Intel will also provide 
the additional proprietary components of the Magma library that need to be 
installed into the same Lava namespace, which enable compiling Lava 
Processes for Loihi systems.

# Licensing and reuse

Lava is open to modification and extension to third-party libraries like Nengo,
ROS, YARP and others. Additional utilities also allow users to profile power and
performance of workloads, visualize complex networks, or help with the float to
fixed point conversions required for many low-precision devices such as
neuromorphic HW.

All of Lava's core APIs and higher-level components are released, by default,
with permissive BSD 3 licenses in order to encourage the broadest possible
community contribution. Lower-level Magma components needed for mapping
processes to neuromorphic backends are generally released with more restrictive
LGPL-2.1 licensing to discourage commercial proprietary forks of these
technologies. The specific components of Magma needed to compile processes
specifically to Intel Loihi chips remains proprietary to Intel and is not
provided through this GitHub site (see below). Similar Magma-layer code for
other future commercial neuromorphic platforms likely will also remain
proprietary.

# Stay in touch

To receive regular updates on the latest developments and releases of the Lava
Software Framework
please [subscribe to our newsletter](http://eepurl.com/hJCyhb).
