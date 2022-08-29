# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
import unittest
from functools import partial

from message_infrastructure import CppMultiProcessing
from message_infrastructure import SharedMemManager
from message_infrastructure import ProcessType
from message_infrastructure import Actor
from message_infrastructure.multiprocessing import MultiProcessing


class Builder():
    def build(self, i):
        print("Builder run build ", i)


def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        builder = kwargs.pop("builder")
        idx = kwargs.pop("idx")
        builder.build(idx)
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


class TestMultiprocessing(unittest.TestCase):
    def test_multiprocessing_spawn(self):
        """
        Spawns an actor.
        Checks that an actor is spawned successfully.
        """
        pass

    def test_multiprocessing_shutdown(self):
        """
        Spawns an actor and sends a stop signal.
        Checks that actor is stopped successfully.
        """
        pass

    def test_multiprocessing_kill(self):
        """
        Spawns an actor and kills it after a certain time.
        Checks that multiprocessing daemon will automatically restart crashed
        actor and reassign to correct process.
        """
        pass

    def test_get_actor_pid(self):
        pass

    def test_get_actor_list(self):
        pass

    def test_get_shared_memory_manager(self):
        pass


def test_multiprocessing():
    mp = MultiProcessing()
    mp.start()
    builder = Builder()
    for i in range(5):
        bound_target_fn = partial(target_fn, idx=i)
        ret = mp.build_actor(bound_target_fn, builder)
        if ret == ProcessType.ChildProcess :
            print("child process, exit")
            exit(0)

    shmm = mp.smm
    for i in range(5):
        print("shared memory id: ", shmm.alloc_mem(8))

    actors = mp.actors
    print(actors)
    print("actor status: ", actors[0].get_status())
    print("stop num: ", shmm.stop())
    print("stop num: ", shmm.stop())

    mp.stop()


test_multiprocessing()
