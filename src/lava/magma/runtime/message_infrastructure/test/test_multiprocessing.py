# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
import unittest
import psutil
import os
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


class TestMultiprocessing(unittest.TestCase):
    mp = MultiProcessing()

    def test_multiprocessing_spawn(self):
        """
        Spawns an actor.
        Checks that an actor is spawned successfully.
        """
        self.mp.start()
        builder = Builder()
        return_type = self.mp.build_actor(target_fn, builder, idx=1)
        # build_actor returns 0 if it is a child process
        self.assertEqual(return_type, 0)

        self.test_get_actor_pid()

    def test_multiprocessing_shutdown(self):
        """
        Spawns an actor and sends a stop signal.
        Checks that actor is stopped successfully.
        """
        self.test_multiprocessing_spawn()

        actor_list = self.mp.actor_pids()
        self.mp.stop()

        # Check that all actor PIDs no longer exist
        for actor_pid in actor_list:
            self.assertFalse(psutil.pid_exists(actor_pid))

    @unittest.skip("Test case for reassigning actors to Processes not yet \
        implemented")
    def test_multiprocessing_kill(self):
        """
        Spawns an actor and kills it after a certain time.
        Checks that multiprocessing daemon will automatically restart crashed
        actor and reassign to correct process.
        """
        self.test_multiprocessing_spawn()

        # Gets list of actor PIDs and kills the 1st one in the list
        actor_list = self.mp.actor_pids()
        os.kill(actor_list[0])

        # TODO: How to check that an actor has been reassinged to the correct
        # process

    def test_get_actor_pid(self):
        """
        Gets list of actor PIDs
        Checks that all actor PIDs exist
        """
        actor_list = self.mp.actor_pids()
        for actor_pid in actor_list:
            self.assertTrue(psutil.pid_exists(actor_pid))

    def test_get_actor_list(self):
        """
        Gets list of actors
        Checks that all actors are of Actor type
        """
        actor_list = self.mp.actors()
        for actor in actor_list:
            self.assertIsInstance(actor, Actor)

    def test_get_shared_memory_manager(self):
        """
        Gets the Shared Memory Manager
        Checks that the shared memory manager is of SharedMemManager type
        """
        shared_memory_manager = self.mp.smm()
        self.assertIsInstance(shared_memory_manager, SharedMemManager)


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
