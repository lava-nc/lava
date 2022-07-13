_Lava_ v0.4.0 brings initial support to compile and run models on Loihi 2 via Intel’s cloud hosted Loihi systems through participation in the Intel Neuromorphic Research Community (INRC). In addition, new tutorials and documentation explain how to build Lava Processes written in Python or C for CPU and Loihi backends (C and Loihi tutorials available via the INRC). 

While this release offers few high-level application examples, Lava v0.4.0 provides major enhancements to the overall Lava architecture. It forms the basis for the open-source community to enable the full Loihi feature set, such as on-chip learning, convolutional connectivity, or accelerated spike IO. The Lava Compiler and Runtime architecture has also been generalized allowing extension to other backends or neuromorphic processors. 

## New Features and Improvements
Features marked with * are available as part of the Loihi 2 extension.
- *Extended Process library including new ProcessModels and additional improvements:
  - LIF, Sigma-Delta, and Dense Processes execute on Loihi NeuroCores.
  - Prototype Convolutional Process added.
  - Sending and receiving spikes to NeuroCores via embedded processes that can be programmed in C with examples included. 
  - All Lava Processes now list all constructor arguments explicitly with type annotations. 
- *Added high-level API to develop custom ProcessModels that use Loihi 2 features:
  - Loihi NeuroCores can be programmed in Python by allocating neural network resources like Axons, Synapses or Neurons. In particular, Loihi 2 NeuroCore Neurons can be configured by writing highly flexible assembly programs.
  - Loihi embedded processors can be programmed in C. But unlike the prior NxSDK, no knowledge of low-level registers details is required anymore. Instead, the C API mirrors the high-level Python API to interact with other processes via channels.
- Compiler and Runtime support for Loihi 2:
  - General redesign of Compiler and Runtime architecture to support compilation of Processes that execute across a heterogenous backend of different compute resources. CPU and Loihi are supported via separate sub compilers.
  - *The Loihi NeuroCore sub compiler automatically distributes neural network resources across multiple cores.
  - *The Runtime supports direct channel-based communication between Processes running on Loihi NeuroCores, embedded CPUs or host CPUs written in Python or C. Of all combinations, only Python<->C and C<->NeuroCore are currently supported.
  - *Added support to access Process Variables on Loihi NeuroCores at runtime via Var.set and Var.get().
- New tutorials and improved class and method docstrings explain how new Lava features can be used such as *NeuroCore and *embedded processor programming.
- An extended suite of unit tests and new *integration tests validate the correctness of the Lava framework.


## Bug Fixes and Other Changes

- Support for virtual ports on multiple incoming connections (Python Processes only) (Issue [#223](https://github.com/lava-nc/lava/issues/223), PR [#224](https://github.com/lava-nc/lava/pull/224))
- Added conda install instructions (PR [#225](https://github.com/lava-nc/lava/pull/225))
- Var.set/get() works when RunContinuous RunMode is used (Issue [#255](https://github.com/lava-nc/lava/issues/255), PR [#256](https://github.com/lava-nc/lava/pull/256)) 
- Successful execution of tutorials now covered by unit tests (Issue [#243](https://github.com/lava-nc/lava/issues/243), PR [#244](https://github.com/lava-nc/lava/pull/244))
- Fixed PYTHONPATH in tutorial_01 (Issue [#45](https://github.com/lava-nc/lava/issues/45), PR [#239](https://github.com/lava-nc/lava/pull/239))
- Fixed output of tutorial_07 (Issue [#249](https://github.com/lava-nc/lava/issues/249), PR [#253](https://github.com/lava-nc/lava/pull/253))

## Breaking Changes

- Process constructors for standard library processes now require explicit keyword/value pairs and do not accept arbitrary input arguments via **kwargs anymore. This might break some workloads.
- use_graded_spike kwarg has been changed to num_message_bits for all the built-in processes.
- shape kwarg has been removed from Dense process. It is automatically inferred from the weight parameter’s shape.
- Conv Process has additional arguments weight_exp and num_weight_bits that are relevant for fixed-point implementations.
- The sign_mode argument in the Dense Process is now an enum rather than an integer.
- New parameters u and v in the LIF Process enable setting initial values for current and voltage.
- The bias parameter in the LIF Process has been renamed to bias_mant.


## Known Issues

- Lava does currently not support on-chip learning, Loihi 1 and a variety of connectivity compression features such as convolutional encoding.
- All Processes in a network must currently be connected via channels. Running unconnected Processes using NcProcessModels in parallel currently gives incorrect results.
- Only one instance of a Process targeting an embedded processor (using CProcessModel) can currently be created. Creating multiple instances in a network, results in an error. As a workaround, the behavior of multiple Processes can be fused into a single CProcessModel.
- Direct channel connections between Processes using a PyProcessModel and NcProcessModel are not supported.
- In the scenario that InputAxons are duplicated across multiple cores and users expect to inject spikes based on the declared port size, then the current implementation leads to buffer overflows and memory corruption.
- Channel communication between PyProcessModels is slow.
- The Lava Compiler is still inefficient and in need of improvement to performance and memory utilization.
- Virtual ports are only supported between Processes using PyProcModels, but not between Processes when CProcModels or NcProcModels are involved. In addition, VirtualPorts do not support concatenation yet.
- Joining and forking of virtual ports is not supported.
- The Monitor Process does currently only support probing of a single Var per Process implemented via a PyProcessModel. The Monitor Process does currently not support probing of Vars mapped to NeuroCores.
- Despite new docstrings, type annotations, and parameter descriptions to most of the public user-facing API, some parts of the code still have limited documentation and are missing type annotations.


## What's Changed
* Virtual ports on multiple incoming connections by @mathisrichter in https://github.com/lava-nc/lava/pull/224
* Add conda install to README by @Tobias-Fischer in https://github.com/lava-nc/lava/pull/225
* PYTHONPATH fix in tutorial by @jlubo in https://github.com/lava-nc/lava/pull/239
* Fix tutorial04_execution.ipynb by @mgkwill in https://github.com/lava-nc/lava/pull/241
* Tutorial tests by @mgkwill in https://github.com/lava-nc/lava/pull/244
* Update README.md remove vlab instructions by @mgkwill in https://github.com/lava-nc/lava/pull/248
* Tutorial bug fix by @PhilippPlank in https://github.com/lava-nc/lava/pull/253
* Fix get set var by @PhilippPlank in https://github.com/lava-nc/lava/pull/256
* Update runtime_service.py by @PhilippPlank in https://github.com/lava-nc/lava/pull/258
* Release/v0.4.0 by @mgkwill in https://github.com/lava-nc/lava/pull/265

## Thanks to our Contributors

- Intel Corporation: All contributing members of the Intel Neuromorphic Computing Lab

### Open-source community: 
- [Tobias-Fischer](https://github.com/Tobias-Fischer), Tobias Fischer 
- [jlubo](https://github.com/jlubo), Jannik Luboeinski

## New Contributors
* @jlubo made their first contribution in https://github.com/lava-nc/lava/pull/239

**Full Changelog**: https://github.com/lava-nc/lava/compare/v0.3.0...v0.4.0
