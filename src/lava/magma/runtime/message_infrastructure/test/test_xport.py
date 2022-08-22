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
    return np.prod(shape) * np.dtype(dtype).itemsize


def channel(data, name='test_channel'):
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


class TestPyPorts(unittest.TestCase):
    data = np.array([1, 2, 3], np.int32)

    def test_inport(self, ports):
        channel_1 = channel(name="channel_1", data=self.data)
        send_port_1 = channel_1.get_send_port()
        recv_port_1 = channel_1.get_recv_port()
        recv_test_port_1 = InPortVectorDense([recv_port_1], 
                                             None, 
                                             self.data.shape,
                                             self.data.dtype)
        
        # print("PyInPortVectorDense.recv", in_port.recv())
        send_test_port_1 = OutPortVectorDense([send_port_1], 
                                              None, 
                                              self.data.shape,
                                              self.data.dtype)

        recv_test_port_1.start()
        send_test_port_1.start()

        send_test_port_1.send(self.data)
        probe_value = recv_test_port_1.probe()

        # probe_value should be True if message reached the PyInPort
        self.assertTrue(probe_value)

        # Get data that reached PyInPort to empty buffer
        _ = recv_test_port_1.recv()
        # Probe PyInPort
        probe_value = recv_test_port_1.probe()

        # probe_value should be False since PyInPort's buffer was emptied
        self.assertFalse(probe_value)

# def test_outport(ports):
#     out_port = OutPortVectorDense(ports)
#     print("PyOutPortVectorDense.send", out_port.send())


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


# main()

if __name__ == '__main__':
    unittest.main()
