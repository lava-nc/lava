# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
from functools import partial

from message_infrastructure import CppMultiProcessing
from message_infrastructure import SharedMemManager
from message_infrastructure import ProcessType
from message_infrastructure import Actor
from message_infrastructure.multiprocessing import MultiProcessing

import time


class Builder():
    def build(self, i):
        print("Builder run build ", i)
        print("sleep 10 s")
        time.sleep(10)
        print("Builder Achieved")


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
        # print("builder", actor.get_status())
        builder.build(idx)
        return 0
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


def test_multiprocessing():
    mp = MultiProcessing()
    mp.start()
    builder = Builder()
    for i in range(5):
        bound_target_fn = partial(target_fn, idx=i)
        ret = mp.build_actor(bound_target_fn, builder)
        print(ret)

    shmm = mp.smm
    for i in range(5):
        print("shared memory id: ", shmm.alloc_mem(8))

    actors = mp.actors
    actor = actors[0]
    print("actor status: ", actor.get_status())
    actor.stop()
    print("actor status: ", actor.get_status())

    print("stop num: ", shmm.stop())
    print("stop num: ", shmm.stop())

    mp.stop()


test_multiprocessing()
