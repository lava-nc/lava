# DDS Talking with ROS2 Example

This example is for testing msg_lib DDS port talking with ROS2 node. The implementation of DDS port in msg_lib does not use any ROS2 libraries.
This example is using [Fast-DDS](https://github.com/eProsima/Fast-DDS/), [CycloneDDS](https://projects.eclipse.org/projects/iot.cyclonedds) and [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html).

The example contains 2 main parts, ROS2 CPP package and Python package.

## Installation of ROS2
Please follow [ROS2 Foxy Installation](https://docs.ros.org/en/foxy/Installation.html) to install ROS2 first.

The defualt middleware that ROS2 uses is [Fast-RTPS](https://fast-dds.docs.eprosima.com/en/v1.7.0/) and the default DDS implementation is [eProsima’s Fast DDS](https://github.com/eProsima/Fast-DDS/). If you want to test and valid msg_lib Fast-DDS port talking with Ros2 node, so please just use the default middleware and DDS version of ROS2.

Or if you want to test msg_lib CycloneDDS port talking with ROS2 node, please follow the [guide](https://docs.ros.org/en/foxy/Installation/DDS-Implementations/Working-with-Eclipse-CycloneDDS.html) to install and enable CycloneDDS rmw for ROS2.

## Build DDSMetadata Package in ROS2 Workspace
As this example need to transfer DDSMetadata type data between ROS2 node and DDS port, DDSMetadata package needs to be built in ROS2 workspace first. Please follow the [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html) to build the DDSMetadata Package.

We have already prepared `DDSMetaData` ROS2 package for the message communication structure between ROS2 and msglib. User can build `DDSMetaData` package by the commands:
```
# open your ros2 build folder and create the ROS2 environment
$ . ~/ros2_foxy/install/local_setup.bash

# build the DDSMetaData ROS2 Package
$ cd <lava>/src/lava/magma/runtime/_c_message_infrastructure/examples/ros2
$ colcon build --packages-select ddsmetadata
```

## Build ROS2 Example Package in ROS2 Workspace
This example also provides the ROS2 package to communicate with DDS port in `ros_talk_with_dds_cpp` and `ros_talk_with_dds_py` folder. You can follow the commands here to build these packages
```
# import the DDSMetaData Package
$ . install/local_setup.bash

# build ros_talk_with_dds_cpp or ros_talk_with_dds_py Package
$ colcon build --packages-select ros_talk_with_dds_cpp
# or
$ colcon build --packages-select ros_talk_with_dds_py
```

For the detail please follow the [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html) to build the cpp code in the folder as a ROS2 package. And for the python code, please follow this [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html).

## Build the DDS Example code
Please follow the `README.md` in path `src/lava/magma/runtime/_c_message_infrastructure/` to build the message infrastrucure library and initialize the environment.

Note : Please change cmake args with the options to choose to build the project with FASTDDS or CycloneDDS.
```
$ export CMAKE_ARGS="-DDDS_CHANNEL=ON -DFASTDDS_ENABLE=ON -DCMAKE_BUILD_TYPE=Debug"
```
or 
```
$ export CMAKE_ARGS="-DDDS_CHANNEL=ON -DCycloneDDS=ON -DCMAKE_BUILD_TYPE=Debug"
```
Then run
```
$ cd <lava>
$ poetry install
```
The DDS example binary will be built in <lava>/build/test. (`test_fastdds_from_ros` and `test_fastdds_to_ros`)

## Running the Example withs FASTDDS
Please open 2 terminals. One is for running ROS2 code and the other is for running DDS port.
### 1st Terminal to Run ROS2 Node
1.  Navigate to your ROS2 workspace folder.
    ```
    $ cd <lava>/src/lava/magma/runtime/_c_message_infrastructure/examples/ros2
    ```
2.  Intialize ROS2 environment.
    ```
    $ source /opt/ros/foxy/setup.bash
    $ . install/local_setup.bash
    ```
3.  Enable Qos configuration in `profile.xml`. [Configuring Fast DDS in ROS 2](https://fast-dds.docs.eprosima.com/en/latest/fastdds/ros2/ros2_configure.html) could be a reference for understanding the QOS configuration for ROS2.

    Run the commands,
    ```
    $ export FASTRTPS_DEFAULT_PROFILES_FILE=<message_infastructure>/examples/ros2/profile.xml

    $ export RMW_FASTRTPS_USE_QOS_FROM_XML=1
    ```

4.  Users could choose to run CPP or Python ROS2 package.

    (1) Run the `publisher`/`subscriber` node of `ros_talk_with_dds_cpp` ROS2 package.
    ```
    $ ros2 run ros_talk_with_dds_cpp ros_pub
    ```
    or
    ```
    $ ros2 run ros_talk_with_dds_cpp ros_sub
    ```
    (2) Run the `publisher`/`subscriber` node of `ros_talk_with_dds_py` ROS2 package.
    ```
    $ ros2 run ros_talk_with_dds_py ros_pub
    ```
    or
    ```
    $ ros2 run ros_talk_with_dds_py ros_sub
    ```
### 2nd Terminal to Run FASTDDS Port
#### Python Example
Users could use Python test files to valid the function of this example.
1.  `test_fastdds_to_ros.py` is corresponding to `subscriber` nodes of `ros_talk_with_dds_cpp` and `ros_talk_with_dds_py` ROS2 package. Please run the command:
    ```
    $ python test_fastdds_to_ros.py
    ```
2.  `test_fastdds_from_ros.py` is corresponding to `publisher` ROS2 node. Please run the command:
    ```
    $ python test_fastdds_from_ros.py
    ```
#### CPP Example
To enbale CPP example, when users are processing the steps in 'Build the DDS Example code' part, the option for build the CPP test example need to be set on, just as,
```
$ cmake .. -DDDS_CHANNEL=ON -DFASTDDS_ENABLE=ON -DCMAKE_BUILD_TYPE=Debug
```
Then the CPP example test binaries will be generated in the folder,
`<lava>/build/test/`
and the names for the tests are `test_fastdds_to_ros` and `test_fastdds_from_ros`.
1.  Navigate to the msg_lib build folder,
    ```
    $ cd <lava>/build/test/
    ```
2.  `test_fastdds_to_ros` is corresponding to `subscriber` node of ROS2 package. Please run the command:
    ```
    $ ./test_fastdds_to_ros
    ```
3.  `test_fastdds_from_ros` is corresponding to `subscriber` node of ROS2 package. Please run the command:
    ```
    $ ./test_fastdds_from_ros
    ```

## Running the Example withs CycloneDDS
Please open 2 terminals. One is for running ROS2 code and the other is for running DDS port.
### 1st Terminal to Run ROS2 Node
1.  Navigate to your ROS2 workspace folder.
    ```
    $ cd <lava>/src/lava/magma/runtime/_c_message_infrastructure/examples/ros2
    ```
2.  Intialize ROS2 environment.
    ```
    $ source /opt/ros/foxy/setup.bash
    $ . install/local_setup.bash
    ```
3.  Enable Cyclone middleware of ROS2 to make ROS2 communication through CycloneDDS. Make sure you have successfully install Cyclone middleware for ROS2. Then run the command in the terminal,
    ```
    $ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ```

4.  Users could choose to run CPP or Python ROS2 package.

    (1) Run the `publisher`/`subscriber` node of `ros_talk_with_dds_cpp` ROS2 package.
    ```
    $ ros2 run ros_talk_with_dds_cpp ros_pub
    ```
    or
    ```
    $ ros2 run ros_talk_with_dds_cpp ros_sub
    ```
    (2) Run the `publisher`/`subscriber` node of `ros_talk_with_dds_py` ROS2 package.
    ```
    $ ros2 run ros_talk_with_dds_py ros_pub
    ```
    or
    ```
    $ ros2 run ros_talk_with_dds_py ros_sub
    ```
### 2nd Terminal to Run CycloneDDS Port
#### Python Example
Users could use Python test files to valid the function of this example.
1.  `test_cyclonedds_to_ros.py` is corresponding to `subscriber` nodes of `ros_talk_with_dds_cpp` and `ros_talk_with_dds_py` ROS2 package. Please run the command:
    ```
    $ python test_cyclonedds_to_ros.py
    ```
2.  `test_cyclonedds_from_ros.py` is corresponding to `publisher` ROS2 node. Please run the command:
    ```
    $ python test_cyclonedds_from_ros.py
    ```
#### CPP Example
To enbale CPP example, when users are processing the steps in 'Build the DDS Example code' part, the option for build the CPP test example need to be set on, just as,
```
$ cmake .. -DDDS_CHANNEL=ON -DCycloneDDS=ON -DCMAKE_BUILD_TYPE=Debug
```
Then the CPP example test binaries will be generated in the folder,
`<lava>/build/test/`
and the names for the tests are `test_cyclonedds_to_ros` and `test_cyclonedds_from_ros`.
1.  Navigate to the msg_lib build folder,
    ```
    $ cd <lava>/build/test/
    ```
2.  `test_cyclonedds_to_ros` is corresponding to `subscriber` node of ROS2 package. Please run the command:
    ```
    $ ./test_cyclonedds_to_ros
    ```
3.  `test_cyclonedds_from_ros` is corresponding to `subscriber` node of ROS2 package. Please run the command:
    ```
    $ ./test_cyclonedds_from_ros
    ```
    