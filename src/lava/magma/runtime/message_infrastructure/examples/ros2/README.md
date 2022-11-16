# DDS Talking with ROS2 Example

This example is for testing msg_lib DDS port talking with ROS2 node. The implementation of DDS port in msg_lib does not use any ROS2 libraries.
This example is using [Fast-DDS](https://github.com/eProsima/Fast-DDS/) and [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) .

## Installation of ROS2
Please follow [ROS2 Foxy Installation](https://docs.ros.org/en/foxy/Installation.html) to install ROS2 first.

The defualt middleware that ROS2 uses is [Fast-RTPS](https://fast-dds.docs.eprosima.com/en/v1.7.0/) and the default DDS implementation is [eProsima’s Fast DDS](https://github.com/eProsima/Fast-DDS/). They are just the implementations this example needs, so please just use the default middleware and DDS version of ROS2.

## Build DDSMetadata Package in ROS2 Workspace
As this example need to transfer DDSMetadata type data between ROS2 node and DDS port, DDSMetadata package needs to be built in ROS2 workspace first. Please follow the [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html) to build the DDSMetadata Package.

Users do not need to define `*.msg` themselves. Please use `DDSMetaData` folder to replace the `DDSMetaData` ROS2 package created by the command,
```
ros2 pkg create --build-type ament_cmake DDSMetaData
```
Then continue following the [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html) to finish the package building.

## Build ROS2 Example Package in ROS2 Workspace
This example also provides the ROS2 package to communicate with DDS port in `ros_talk_with_dds` folder. Please follow the [guide](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html) to build the code in the folder as a ROS2 package.
## Build the DDS Example code
Please follow the `README.md` in path `src/lava/magma/runtime/message_infrastructure/` to build the message infrastrucure library and initial the environment.

Note : Please run the `cmake` command with the options,
```
cmake .. -DDDS_CHANNEL=ON -DFASTDDS_ENABLE=ON
```
## Running the Example
Please open 2 terminals. One is for running ROS2 code and the other is for running DDS port.
### 1st Terminal to Run ROS2 Node
1. Navigate to the your ROS2 workspace folder. For example,
    ```
    $ cd ~/ros2_ws
    ```
2. Intialize ROS2 env.
    ```
    $ source /opt/ros/foxy/setup.bash
    $ . install/local_setup.bash
    ```
3. Enable Qos configuration in `ros_talk_with_dds/profile.xml`. [Configuring Fast DDS in ROS 2](https://fast-dds.docs.eprosima.com/en/latest/fastdds/ros2/ros2_configure.html) could be a reference for understanding the QOS configuration for ROS2.

    Run the commands,
    ```
    $ export FASTRTPS_DEFAULT_PROFILES_FILE=<lava_repo>/src/lava/magma/runtime/message_infrastructure/examples/ros2/ros_talk_with_dds/profile.xml

    $ export RMW_FASTRTPS_USE_QOS_FROM_XML=1
    ```

4. Run the `publisher`/`subscriber` node of `ros_talk_with_dds` ROS2 package.
    ```
    $ ros2 run ros_talk_with_dds ros_pub
    ```
    or
    ```
    $ ros2 run ros_talk_with_dds ros_sub
    ```
### 2nd Terminal to Run DDS Port
#### Python Example
Users could use Python test files to valid the function of this example.
1. `test_dds_to_ros.py` is corresponding to `subscriber` node of `ros_talk_with_dds` ROS2 package. Please run the command:
    ```
    python test_dds_to_ros.py
    ```
2. `test_ros_to_dds.py` is corresponding to `publisher` node of `ros_talk_with_dds` ROS2 package. Please run the command:
    ```
    python test_ros_to_dds.py
    ```
#### CPP Example
To enbale CPP example, when users are processing the steps in 'Build the DDS Example code' part, the option for build the CPP test example need to be set on, just as,
```
cmake .. -DDDS_CHANNEL=ON -DFASTDDS_ENABLE=ON -DCMAKE_BUILD_TYPE=Debug
```
Then the CPP example test binaries will be generated in the folder,
`<lava_repo>/src/lava/magma/runtime/message_infrastructure/build/test`
and the names for the tests are `test_dds_to_ros` and `test_ros_to_dds`.
1. Navigate to the msg_lib build folder,
    ```
    cd <lava_repo>/src/lava/magma/runtime/message_infrastructure/build
    ```
2. `test_dds_to_ros` is corresponding to `subscriber` node of `ros_talk_with_dds` ROS2 package. Please run the command:
    ```
    test/test_dds_to_ros
    ```
3. `test_ros_to_dds` is corresponding to `subscriber` node of `ros_talk_with_dds` ROS2 package. Please run the command:
    ```
    test/test_ros_to_dds
    ```