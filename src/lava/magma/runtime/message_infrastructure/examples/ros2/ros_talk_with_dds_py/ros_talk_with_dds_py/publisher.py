# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from ddsmetadata.msg import DDSMetaData

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        ros_qos = qos_profile_system_default
        self.publisher_ = self.create_publisher(DDSMetaData, 'dds_topic', ros_qos)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.msg = DDSMetaData()
        self.msg.nd = 1
        self.msg.type = 7
        self.msg.elsize = 8
        self.msg.total_size = 1
        self.msg.dims[0] = 1
        self.msg.strides[0] = 1

    def timer_callback(self):
        mdata = [self.i % 255]
        self.msg.mdata = mdata
        self.publisher_.publish(self.msg)
        self.get_logger().info('Publishing: "%ld"' % self.msg.mdata[0])
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
