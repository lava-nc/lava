# Release 0.3.0

Lava 0.3.0 includes bug fixes, updated documentation, improved error handling, refactoring of the Lava Runtime and support for sigma delta neuron enconding and decoding.

## Features and Improvements

* Added sigma delta neuron enconding and decoding support ([PR #180](https://github.com/lava-nc/lava/pull/180), [Issue #179](https://github.com/lava-nc/lava/issues/179))
* Implementation of ReadVar and ResetVar IO process ([PR #156](https://github.com/lava-nc/lava/pull/156), [Issue #155](https://github.com/lava-nc/lava/issues/155))
* Added Runtime handling of exceptions occuring in ProcessModels and the Runtime now returns exeception stack traces ([PR #135](https://github.com/lava-nc/lava/pull/135), [Issue #83](https://github.com/lava-nc/lava/issues/83))
* Virtual ports for reshaping and transposing (permuting) are now supported. ([PR #187](https://github.com/lava-nc/lava/pull/187), [Issue #185](https://github.com/lava-nc/lava/issues/185), [PR #195](https://github.com/lava-nc/lava/pull/195), [Issue #194](https://github.com/lava-nc/lava/issues/194))
* A Ternary-LIF neuron model was added to the process library. This new variant supports both positive and negative threshold for processing of signed signals ([PR #151](https://github.com/lava-nc/lava/pull/151), [Issue #150](https://github.com/lava-nc/lava/issues/150))
* Refactored runtime to reduce the number of channels used for communication([PR #157](https://github.com/lava-nc/lava/pull/157), [Issue #86](https://github.com/lava-nc/lava/issues/86))
* Refactored Runtime to follow a state machine model and refactored ProcessModels to use command design pattern, implemented PAUSE and RUN CONTINOUS ([PR #180](https://github.com/lava-nc/lava/pull/171), [Issue #86](https://github.com/lava-nc/lava/issues/86), [Issue #52](https://github.com/lava-nc/lava/issues/53))
* Refactored builder to its own package ([PR #170](https://github.com/lava-nc/lava/pull/170), [Issue #169](https://github.com/lava-nc/lava/issues/169))
* Refactored PyPorts implementation to fix incomplete PyPort hierarchy ([PR #131](https://github.com/lava-nc/lava/pull/131), [Issue #84](https://github.com/lava-nc/lava/issues/84))
* Added improvements to the MNIST tutorial ([PR #147](https://github.com/lava-nc/lava/pull/147), [Issue #146](https://github.com/lava-nc/lava/issues/146))
* A standardized template is now in use on new Pull Requests and Issues ([PR #140](https://github.com/lava-nc/lava/pull/140))
* Support added for editable install ([PR #93](https://github.com/lava-nc/lava/pull/93), [Issue #19](https://github.com/lava-nc/lava/issues/19))
* Improved runtime documentation ([PR #167](https://github.com/lava-nc/lava/pull/167))

## Bug Fixes and Other Changes
* Fixed multiple Monitor related issues ([PR #128](https://github.com/lava-nc/lava/pull/128), [Issue #103](https://github.com/lava-nc/lava/issues/103), [Issue #104](https://github.com/lava-nc/lava/issues/104), [Issue #116](https://github.com/lava-nc/lava/issues/116), [Issue #127](https://github.com/lava-nc/lava/issues/127))
* Fixed packaging issue regarding the dataloader for MNIST ([PR #133](https://github.com/lava-nc/lava/pull/133))
* Fixed multiprocessing bug by checking process lineage before join ([PR #177](https://github.com/lava-nc/lava/pull/177), [Issue #176](https://github.com/lava-nc/lava/issues/176))
* Fixed priority of channel commands in model ([PR #190](https://github.com/lava-nc/lava/pull/190), [Issue #186](https://github.com/lava-nc/lava/issues/186))
* Fixed RefPort time step handling ([PR #205](https://github.com/lava-nc/lava/pull/205), [Issue #204](https://github.com/lava-nc/lava/issues/204))

## Known Issues
* No support for Intel Loihi
* CSP channels process communication, implemented with Python multiprocessing, needs improvement to reduce the overhead from inter-process communication to approach native execution speeds of similar implementations without CSP channel overhead
* Virtual ports for concatenation are not supported
* Joining and forking of virtual ports is not supported
* A Monitor process cannot monitor more than one Var/InPort of a process, as a result multi-var probing with a singular Monitor process is not supported
* Limited API documentation

## What's Changed
* Fixing multiple small issues of the Monitor proc by @elvinhajizada in https://github.com/lava-nc/lava/pull/128
* GitHub Issue/Pull request template by @mgkwill in https://github.com/lava-nc/lava/pull/140
* Fixing MNIST dataloader by @tihbe in https://github.com/lava-nc/lava/pull/133
* Runtime error handling by @PhilippPlank in https://github.com/lava-nc/lava/pull/135
* Reduced the number of channels between service and process (#1) by @ysingh7 in https://github.com/lava-nc/lava/pull/157
* TernaryLIF and refactoring of LIF to inherit from AbstractLIF by @srrisbud in https://github.com/lava-nc/lava/pull/151
* Proc_params for communicating arbitrary object between process and process model by @bamsumit in https://github.com/lava-nc/lava/pull/162
* Support editable install by @matham in https://github.com/lava-nc/lava/pull/93
* Implementation of ReadVar and ResetVar IO process and bugfixes for LIF, Dense and Conv processes by @bamsumit in https://github.com/lava-nc/lava/pull/156
* Refactor builder to module by @mgkwill in https://github.com/lava-nc/lava/pull/170
* Use unittest ci by @mgkwill in https://github.com/lava-nc/lava/pull/173
* Improve mnist tutorial by @srrisbud in https://github.com/lava-nc/lava/pull/147
* Multiproc bug by @mgkwill in https://github.com/lava-nc/lava/pull/177
* Refactoring py/ports by @PhilippPlank in https://github.com/lava-nc/lava/pull/131
* Adds runtime documentation by @joyeshmishra in https://github.com/lava-nc/lava/pull/167
* Implementation of Pause and Run Continuous with refactoring of Runtime by @ysingh7 in https://github.com/lava-nc/lava/pull/171
* Ref port debug by @PhilippPlank in https://github.com/lava-nc/lava/pull/183
* Sigma delta neuron, encoding and decoding support by @bamsumit in https://github.com/lava-nc/lava/pull/180
* Add NxSDKRuntimeService by @mgkwill in https://github.com/lava-nc/lava/pull/182
* Partial implementation of virtual ports for PyProcModels by @mathisrichter in https://github.com/lava-nc/lava/pull/187
* Remove old runtime_service.py by @mgkwill in https://github.com/lava-nc/lava/pull/192
* Fixing priority of channel commands in model by @PhilippPlank in https://github.com/lava-nc/lava/pull/190
* Virtual ports between RefPorts and VarPorts by @mathisrichter in https://github.com/lava-nc/lava/pull/195
* RefPort's sometimes handled a time step late by @PhilippPlank in https://github.com/lava-nc/lava/pull/205
* Fixed reset timing offset by @bamsumit in https://github.com/lava-nc/lava/pull/207
* Update README.md by @mgkwill in https://github.com/lava-nc/lava/pull/202

## New Contributors
* @tihbe made their first contribution in https://github.com/lava-nc/lava/pull/133
* @ysingh7 made their first contribution in https://github.com/lava-nc/lava/pull/157
* @matham made their first contribution in https://github.com/lava-nc/lava/pull/93

**Full Changelog**: https://github.com/lava-nc/lava/compare/v0.2.0...v0.3.0
