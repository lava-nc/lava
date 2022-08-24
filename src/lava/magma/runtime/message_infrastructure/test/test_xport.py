# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from message_infrastructure import get_channel_factory
from message_infrastructure import SharedMemory
from message_infrastructure import ChannelTransferType
from MessageInfrastructurePywrapper import (
    InPortVectorDense,
    OutPortVectorDense,
    VarPortVectorSparse,
)


def nbytes_cal(shape, dtype):
    """Calculate number of bytes (size) of data to be sent/received"""
    return np.prod(shape) * np.dtype(dtype).itemsize


def get_channel(data, name):
    channel_factory = get_channel_factory()
    shm = SharedMemory()
    size = 2
    nbytes = nbytes_cal(data.shape, data.dtype)
    return channel_factory.get_channel(ChannelTransferType.SHMEMCHANNEL,
                                       shm,
                                       data,
                                       size,
                                       nbytes,
                                       name)


def create_channel(data, channel_name="test_channel"):
    """
    Creates a channel and returns a pair of send and receive ports connected
    to the channel.

    Parameters
    ----------
    data [NumPy array] : Data to be sent, used to determine data type and size
    channel_name [str] : Name of channel

    Returns
    -------
    send_port : Sending port connected to channel
    recv_port : Receiving port connected to channel
    """
    channel = get_channel(data, channel_name)
    send_port = channel.get_send_port()
    receive_port = channel.get_recv_port()

    return send_port, receive_port


class TestPorts(unittest.TestCase):
    """Test ports functionalities"""

    data = np.array([1, 2, 3], np.int32)

    def test_send_and_receive_intergers(self):
        """
        Test that InPort received integer data that is send by OutPort.
        """
        sent_data = np.array([1, 2, 3], np.int32)
        send_port, recv_port = create_channel(self.data,
                                              channel_name="int_channel")

        # Create an InPort
        in_port = InPortVectorDense([recv_port], None,
                                    self.data.shape, self.data.dtype)

        # Create an OutPort
        out_port = OutPortVectorDense([send_port], None,
                                      self.data.shape, self.data.dtype)

        # Initialize InPort and OutPort
        in_port.start()
        out_port.start()

        # Send data through OutPort
        out_port.send(self.data)

        # Get received data
        received_data = in_port.recv()
        self.assertEqual(received_data, sent_data)

    def test_send_and_receive_floats(self):
        """
        Test that InPort received floating-point data that is send by OutPort.
        """
        sent_data = np.array([3.14, 13.89, 0.54], np.float32)
        send_port, recv_port = create_channel(self.data,
                                              channel_name="float_channel")

        # Create an InPort
        in_port = InPortVectorDense([recv_port], None,
                                    self.data.shape, self.data.dtype)

        # Create an OutPort
        out_port = OutPortVectorDense([send_port], None,
                                      self.data.shape, self.data.dtype)

        # Initialize InPort and OutPort
        in_port.start()
        out_port.start()

        # Send data through OutPort
        out_port.send(self.data)

        # Get received data
        received_data = in_port.recv()
        self.assertEqual(received_data, sent_data)

    def test_probe(self):
        """
        Tests that probe() returns True when InPort buffer has content,
        and returns False when InPort buffer is empty.
        """
        # Creating a pair of send and receive ports from channel
        send_port_1, recv_port_1 = create_channel(self.data,
                                                  channel_name="Channel_1")

        # Create an InPort
        recv_test_port_1 = InPortVectorDense([recv_port_1],
                                             None,
                                             self.data.shape,
                                             self.data.dtype)

        # Create an OutPort
        send_test_port_1 = OutPortVectorDense([send_port_1],
                                              None,
                                              self.data.shape,
                                              self.data.dtype)

        # Initialize InPort and OutPort
        recv_test_port_1.start()
        send_test_port_1.start()

        # Send data through OutPort
        send_test_port_1.send(self.data)

        # Probe that InPort has received the data (boolean)
        probe_value = recv_test_port_1.probe()

        # probe_value should be True if message reached the InPort
        self.assertTrue(probe_value)

        # Get data that reached InPort to empty buffer
        _ = recv_test_port_1.recv()

        # Probe InPort
        probe_value = recv_test_port_1.probe()

        # probe_value should be False since InPort's buffer was emptied
        self.assertFalse(probe_value)


# def test_varport(s_ports, r_ports):
#     var_port = VarPortVectorSparse("test", s_ports, r_ports)
#     print("PyVarPortVectorSparse.service", var_port.service())

# def main():
#     channels = [channel() for _ in range(3)]

#     send_ports = [c.get_send_port() for c in channels]
#     recv_ports = [c.get_recv_port() for c in channels]

#     test_inport(recv_ports)
#     test_outport(send_ports)
#     test_varport(send_ports, recv_ports)

#     print("finish test function.")


# Run unit tests
if __name__ == '__main__':
    unittest.main()
