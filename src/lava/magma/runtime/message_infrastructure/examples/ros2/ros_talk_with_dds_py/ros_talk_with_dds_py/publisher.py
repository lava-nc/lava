# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from ddsmetadata.msg import DDSMetaData
from .utils.np_mdata_trans import nparray_to_metadata


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(DDSMetaData,
                                                'dds_topic',
                                                qos_profile_system_default)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        np_arr = np.array(([self.i, 2, 3]), np.int64)
        msg = nparray_to_metadata(np_arr)
        self.publisher_.publish(msg)
        print("Publishing : ", np_arr)
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
