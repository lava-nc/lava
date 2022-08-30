# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from message_infrastructure import get_channel_factory
from message_infrastructure import SharedMemory
from message_infrastructure import ChannelBackend
from MessageInfrastructurePywrapper import (
    InPortVectorDense,
    OutPortVectorDense,
    VarPortVectorSparse,
)


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def channel():
    channel_factory = get_channel_factory()
    data = np.array([1, 2, 3], np.int32)
    shm = SharedMemory()
    size = 2
    nbytes = nbytes_cal(data.shape, data.dtype)
    name = 'test_channel'
    return channel_factory.get_channel(ChannelBackend.SHMEMCHANNEL,
                                       shm,
                                       data,
                                       size,
                                       nbytes,
                                       name)


def test_inport(ports):
    in_port = InPortVectorDense(ports)
    print("PyInPortVectorDense.recv", in_port.recv())


def test_outport(ports):
    out_port = OutPortVectorDense(ports)
    print("PyOutPortVectorDense.send", out_port.send())


def test_varport(s_ports, r_ports):
    var_port = VarPortVectorSparse("test", s_ports, r_ports)
    print("PyVarPortVectorSparse.service", var_port.service())


def main():
    channels = [channel() for _ in range(3)]

    send_ports = [c.get_send_port() for c in channels]
    recv_ports = [c.get_recv_port() for c in channels]

    test_inport(recv_ports)
    test_outport(send_ports)
    test_varport(send_ports, recv_ports)

    print("finish test function.")


main()
