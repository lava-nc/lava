# Release 0.1.0

This first release of Lava introduces its high-level, hardware-agnostic API for developing algorithms of distributed, parallel, and asynchronous processes that communicate with each other via message-passing over channels with each other. The API is released together with the Lava compiler and runtime which together form the Magma layer of the Lava software framework. 

Our initial version of Magma allows you to familiarize yourself with the Lava user interface and to build first algorithms in Python that can be executed on CPU without requiring access to physical or cloud based Loihi resources.

# New Features and Improvements

* New Lava API to build networks of interacting Lava processes 
* New Lava Compiler to map Lava processes to executable Python code for CPU execution (support for Intel Loihi will follow)
* New Lava Runtime to execute Lave processes
* A range of fundamental tutorials illustrating the basic concepts of Lava

# Bug Fixes and Other Changes

* This is the first release of Lava. No bug fixes or other changes

# Thanks to our Contributors

@GaboFGuerra, @joyeshmishra, @PhilippPlank, @drager-intel, @mathisrichter, @srrisbud, @ysingh7, @phstratmann, @mgkwill, @awintel 
  
# Breaking Changes

* This is the first release of Lava. No breaking or other changes. 

# Known Issues

* No support for Intel Loihi yet
* Multiprocessing and channel-based communication not very performant yet
* Virtual ports for reshaping and concatenation are not supported yet
* No support for direct memory access via RefPorts yet
* Connectivity from one to many or from many to one port not supported yet
* No support for live state monitoring yet
* Still limited API documentation
