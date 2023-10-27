# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from ddsmetadata.msg import DDSMetaData
from .utils.np_mdata_trans import metadata_to_nparray


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            DDSMetaData,
            'dds_topic',
            self.listener_callback,
            qos_profile_system_default)
        # pylint: disable=W0104
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        print("Heard : ", metadata_to_nparray(msg))


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
