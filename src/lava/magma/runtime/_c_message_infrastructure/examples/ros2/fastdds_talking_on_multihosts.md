# FASTDDS Talking on Multi-hosts

This doc is the guide for users to test fastdds channel communication on multi-hosts.

Please follow the [guide](https://fast-dds.docs.eprosima.com/en/latest/fastdds/discovery/simple.html) and you could implement your own applications and create your own `profile.xml` files to enbale multi-host communication.

As device limit, this example has only been tested on 2 docker env on a same host machine.

## Example

There are two profiles, one for data sender and the other for reciever.

After enabling lava env with FASTDDS_Backend on 2 machines following `README.md`, modify `reciever_profile.xml` and `sender_profile.xml` to make the `locator` ip address and port aligned with another machine's ip and port.

Then, please enable `reciever_profile.xml` on the machine which will be data reciever.

```
    $ export FASTRTPS_DEFAULT_PROFILES_FILE=profile.xml

    $ export RMW_FASTRTPS_USE_QOS_FROM_XML=1
```
Then run the test,

```
    $ python test_fastdds_from_ros.py
```

And enable `sender_profile.xml` on the machine which will be data sender.


```
    $ export FASTRTPS_DEFAULT_PROFILES_FILE=profile.xml

    $ export RMW_FASTRTPS_USE_QOS_FROM_XML=1
```

Then run the test,

```
    $ python test_fastdds_to_ros.py
```
