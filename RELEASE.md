# Release 0.2.0

Lava 0.2.0 includes several improvements to the Lava Runtime. One of them improves the performance of the underlying message passing framework by over 10x on CPU. We also added new floating-point and Loihi fixed-point PyProcessModels for LIF and DENSE Processes as well as a new CONV Process. In addition, Lava now supports remote memory access between Processes via RefPorts which allows Processes to reconfigure other Processes. Finally, we added/updated several new tutorials to address all these new features.

## Features and Improvements

* Refactored Runtime and RuntimeService to separate the MessagePassingBackend from the Runtime and RuntimeService itself into its own standalone module. This will allow implementing and comparing the performance of other implementations for channel-based communication and also will enable true multi-node scaling beyond the capabilities of the Python multiprocessing module (PR #29)
* Enhanced execution performance by removing busy waits in the Runtime and RuntimeService (Issue #36 & PR #87)
* Enabled compiler and runtime support for RefPorts which allows remote memory access between Lava processes such that one process can reconfigure another process at runtime. Also, remote-memory access is based on channel-based message passing but can lead to side effects and should therefore be used with caution. See Remote Memory Access tutorial for how RefPorts can be used (Issue #43 & PR #46).
* Implemented a first prototype of a Monitor Process. A Monitor provides a user interface to probe Vars and OutPorts of other Processes and records their evolution over time in a time series for post-processing. The current Monitor prototype is limited in that it can only probe a single Var or OutPort per Process. (Issue #74 & PR #80). This limitation will be addressed in the next release.
* Added floating point and Loihi-fixed point PyProcessModels for LIF and connection processes like DENSE and CONV. See issue #40 for more details.
* Added an in-depth tutorial on connecting processes (PR #105)
* Added an in-depth tutorial on remote memory access (PR #99)
* Added an in-depth tutorial on hierarchical Processes and SubProcessModels ()

## Bug Fixes and Other Changes
* Fixed a bug in get/set Var to enable get/set of floating-point values (Issue #44)
* Fixed install instructions (setting PYTHONPATH) (Issue #45)
* Fixed code example in documentation (Issue #62)
* Fixed and added missing license information (Issue #41 & Issue #63)
* Added unit tests for merging and branching In-/OutPorts (PR #106)

## Known Issues
* No support for Intel Loihi yet.
* Channel-based Process communication via CSP channels implemented with Python multiprocessing improved significantly by >30x . However, more improvement is still needed to reduce the overhead from inter-process communication in implementing CSP channels in SW and to get closer to native execution speeds of similar implementations without CSP channel overhead.
* Errors from remote system processes like PyProcessModels or the PyRuntimeService are currently not thrown to the user system process. This makes debugging of parallel processes hard. We are working on propagating exceptions thrown in remote processes to the user.
* Virtual ports for reshaping and concatenation are not supported yet.
* A single Monitor process cannot monitor more than one Var/InPort of single process, i.e., multi-var probing with single Monitor process is not supported yet.
* Still limited API documentation.
* Non-blocking execution mode not yet supported. Thus Runtime.pause() and Runtime.wait() do not work yet.


## What's Changed
* Remove unused channel_utils by @mgkwill in https://github.com/lava-nc/lava/pull/37
* Refactor Message Infrastructure by @joyeshmishra in https://github.com/lava-nc/lava/pull/29
* Fixed copyright in BSD-3 LICENSE files by @mathisrichter in https://github.com/lava-nc/lava/pull/42
* Fixed PYTHONPATH installation instructions after directory restructure of core lava repo by @drager-intel in https://github.com/lava-nc/lava/pull/48
* Add missing license in utils folder by @Tobias-Fischer in https://github.com/lava-nc/lava/pull/58
* Add auto Runtime.stop() by @mgkwill in https://github.com/lava-nc/lava/pull/38
* Enablement of RefPort to Var/VarPort connections by @PhilippPlank in https://github.com/lava-nc/lava/pull/46
* Support float data type for get/set value of Var  by @PhilippPlank in https://github.com/lava-nc/lava/pull/69
* Disable non-blocking execution by @PhilippPlank in https://github.com/lava-nc/lava/pull/67
* LIF ProcessModels: Floating and fixed point: PR attempt #2 by @srrisbud in https://github.com/lava-nc/lava/pull/70
* Fixed bug in README.md example code by @mathisrichter in https://github.com/lava-nc/lava/pull/61
* PyInPort: probe() implementation by @gkarray in https://github.com/lava-nc/lava/pull/77
* Performance improvements by @harryliu-intel in https://github.com/lava-nc/lava/pull/87
* Clean up of explicit namespace declaration by @bamsumit in https://github.com/lava-nc/lava/pull/98
* Enabling monitoring/probing of Vars and OutPorts of processes with Monitor Process by @elvinhajizada in https://github.com/lava-nc/lava/pull/80
* Conv Process Implementation by @bamsumit in https://github.com/lava-nc/lava/pull/73
* Move tutorials to root directory of the repo by @bamsumit in https://github.com/lava-nc/lava/pull/102
* Tutorial for shared memory access (RefPorts) by @PhilippPlank in https://github.com/lava-nc/lava/pull/99
* Move tutorial07 by @PhilippPlank in https://github.com/lava-nc/lava/pull/107
* Added Unit tests for branching/merging of IO ports by @PhilippPlank in https://github.com/lava-nc/lava/pull/106
* Connection tutorial finished by @PhilippPlank in https://github.com/lava-nc/lava/pull/105
* Fix for issue #109, Monitor unit test failing non-deterministically by @mathisrichter in https://github.com/lava-nc/lava/pull/110
* Created floating pt and bit accurate Dense ProcModels + unit tests. Fixes issues #100 and #111. by @drager-intel in https://github.com/lava-nc/lava/pull/112
* Update test_io_ports.py by @PhilippPlank in https://github.com/lava-nc/lava/pull/113
* Fix README.md Example Code by @mgkwill in https://github.com/lava-nc/lava/pull/94
* Added empty list attribute `tags` to `AbstractProcessModel` by @srrisbud in https://github.com/lava-nc/lava/pull/96
* Lava 0.2.0 by @mgkwill in https://github.com/lava-nc/lava/pull/117

## New Contributors
* @joyeshmishra made their first contribution in https://github.com/lava-nc/lava/pull/29
* @drager-intel made their first contribution in https://github.com/lava-nc/lava/pull/48
* @Tobias-Fischer made their first contribution in https://github.com/lava-nc/lava/pull/58
* @PhilippPlank made their first contribution in https://github.com/lava-nc/lava/pull/46
* @gkarray made their first contribution in https://github.com/lava-nc/lava/pull/77
* @harryliu-intel made their first contribution in https://github.com/lava-nc/lava/pull/87
* @bamsumit made their first contribution in https://github.com/lava-nc/lava/pull/98
* @elvinhajizada made their first contribution in https://github.com/lava-nc/lava/pull/80

**Full Changelog**: https://github.com/lava-nc/lava/compare/v0.1.1...v0.2.0
