# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
import unittest
import psutil
import os
import time
import numpy as np
from functools import partial
from enum import Enum

from message_infrastructure import CppMultiProcessing
from message_infrastructure import ProcessType
from message_infrastructure import Actor
from message_infrastructure import ActorStatus
from message_infrastructure.multiprocessing import MultiProcessing
from message_infrastructure import SendPort
from message_infrastructure import RecvPort
from message_infrastructure import Channel

import time


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


class Builder():
    def build(self, i):
        time.sleep(0.0001)


def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        actor = args[0]
        builder = kwargs.pop("builder")
        idx = kwargs.pop("idx")
        builder.build(idx)
        return 0
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


class TestMultiprocessing(unittest.TestCase):

    def test_multiprocessing_actors(self):
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            ret = mp.build_actor(bound_target_fn, builder)

        time.sleep(0.1)
        mp.stop(True)


if __name__ == '__main__':
    unittest.main()
