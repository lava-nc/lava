![image](https://user-images.githubusercontent.com/68661711/135301797-400e163d-71a3-45f8-b35f-e849e8c74f0c.png)
<p align="center"><b>
  A Software Framework for Neuromorphic Computing
</b></p>

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/40792fa7db054279bdf7532e36f0cfab)](https://app.codacy.com/gh/lava-nc/lava/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/40792fa7db054279bdf7532e36f0cfab)](https://app.codacy.com/gh/lava-nc/lava/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)

If you like lava and want to support it, the easiest way is to star our repo (click star in the upper right corner).

# Overview

Lava is an open source SW framework to develop applications for
neuromorphic hardware architectures. It provides developers with the abstractions
and tools to develop distributed and massively parallel applications. These
applications can be deployed to heterogeneous system architectures containing
conventional processors as well as neuromorphic chips that exploit event-based
message passing for communication. The Lava framework comprises high-level
libraries for deep learning, constrained optimization, and others for productive
algorithm development. It also includes tools to map those algorithms to
different types of hardware architectures.


<p align="center">
<img src="https://user-images.githubusercontent.com/68661711/135412508-4a93e20a-8b64-4723-a69b-de8f4b5902f7.png" alt="Lava organization" width="500"/>
</p>

Today Lava supports conventional CPUs and Intel's Loihi architecture, but
its compiler and runtime are open to extension for other architectures.

To learn more about the Lava Software Framework, please refer to the
detailed documentation at http://lava-nc.org/.

The Lava framework is licensed with permissive open source
BSD 3 licensing to highly encourage community contributions.
Lower level components in Lava, that map algorithms to different
hardware backends, are licensed with the LGPL-2.1 license to discourage
commercial proprietary forks. Specific sensitive components
supporting architectures like Intel Loihi may remain proprietary
to Intel and will be shared as extensions to eligible users.

>### Lava extension for Intel's Loihi
>The Lava extension for Loihi is available for members of the Intel Neuromorphic Research Community (INRC). The extension enables execution of Lava on Intel's Loihi hardware platform.
>
>Developers interested in using Lava with Loihi systems need to join the INRC. Loihi 1 and 2 research systems are currently not available commercially. Once a member of the INRC, developers will gain access to cloud-hosted Loihi systems or may be able to obtain physical Loihi systems on a loan basis.
>
>To join the INRC, visit [http://neuromorphic.intel.com](http://neuromorphic.intel.com) or email at [inrc_interest@intel.com](mailto:inrc_interest@intel.com).
>
> If you are already a member of the INRC, please read how to [get started with the Lava extension for Loihi](https://intel-ncl.atlassian.net/wiki/spaces/NAP/pages/1785856001/Get+started+with+the+Lava+extension+for+Loihi). This page is **only** accessible to members of the INRC.


# Getting started
The open-source Lava Software framework and its complementary algorithm
libraries are hosted at [http://github.com/lava-nc](http://github.com/lava-nc) and
the framework supports at minimimum CPU backends.

Note that you should install the core Lava repository [lava](http://github.com/lava-nc/lava)
before installing other Lava libraries such as [lava-optimization](http://github.com/lava-nc/lava-optimization)
or [lava-dl](http://github.com/lava-nc/lava-dl).

## Installing Lava from source

If you are interested in developing in Lava and modifying Lava source code,
we recommend cloning the repository and using `poetry` to setup Lava. You
will need to install the `poetry` Python package.

Open a **python 3** terminal and run based on the OS you are on:

### Linux/MacOS

```bash
cd $HOME
curl -sSL https://install.python-poetry.org | python3 -
git clone git@github.com:lava-nc/lava.git
cd lava
git checkout v0.9.0
./utils/githook/install-hook.sh
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
pytest

## See FAQ for more info: https://github.com/lava-nc/lava/wiki/Frequently-Asked-Questions-(FAQ)#install
```

### Windows

```powershell
# Commands using PowerShell
cd $HOME
git clone git@github.com:lava-nc/lava.git
cd lava
git checkout v0.9.0
python3 -m venv .venv
.venv\Scripts\activate
pip install -U pip
curl -sSL https://install.python-poetry.org | python3 -
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

## Alternative: Installing Lava via Conda

If you use the Conda package manager, you can simply install the Lava package
via:

```bash
conda install lava -c conda-forge
```

Alternatively with intel numpy and scipy:

```bash
conda create -n lava python=3.9 -c intel
conda activate lava
conda install -n lava -c intel numpy scipy
conda install -n lava -c conda-forge lava --freeze-installed
```

## Alternative: Installing Lava from pypi

If you would like to install Lava as a user you can install via pypi binaries.
Installing in this way does not give you access to run tests.

Open a Python terminal and run:

### Windows/MacOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate ## Or Windows: .venv\Scripts\activate
pip install -U pip
pip install lava-nc
```

## Alternative: Installing Lava from binaries

You can also install Lava as a user with published Lava releases via
[GitHub Releases](https://github.com/lava-nc/lava/releases). Please download
the package and install it with the following commands. Installing in this way does not
give you access to run tests.

Open a Python terminal and run:

### Windows/MacOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate ## Or Windows: .venv\Scripts\activate
pip install -U pip
# Substitute lava version needed for lava-nc-<version here>.tar.gz below
pip install lava-nc-0.9.0.tar.gz
```

## Linting, testing, documentation and packaging

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
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
>Refer to the tutorials directory for in-depth as well as end-to-end tutorials on how to write Lava Processes, connect them, and execute the code.

# Stay in touch

To receive regular updates on the latest developments and releases of the Lava
Software Framework
please [subscribe to our newsletter](http://eepurl.com/hJCyhb).
