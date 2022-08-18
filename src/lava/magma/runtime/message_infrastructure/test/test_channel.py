# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from enum import Enum

from message_infrastructure import ShmemChannel
from message_infrastructure import SendPortProxy
from message_infrastructure import RecvPortProxy
from message_infrastructure import ChannelFactory, get_channel_factory
from message_infrastructure import SharedMemory

def nbytes_cal(shape, dtype):
  return np.prod(shape) * np.dtype(dtype).itemsize

class ChannelType(Enum):
  SHMEMCHANNEL = 0
  RPCCHANNEL = 1
  DDSCHANNEL = 2

def main():
  channel_factory = get_channel_factory()
  data = np.ones((2, 2, 2))
  shm = SharedMemory()
  size = 2
  nbytes = nbytes_cal(data.shape, data.dtype)

  shmem_channel = channel_factory.get_channel(ChannelType.SHMEMCHANNEL,
                                              shm,
                                              data,
                                              size,
                                              nbytes)

  send_port = shmem_channel.get_send_port()
  recv_port = shmem_channel.get_recv_port()

main()
