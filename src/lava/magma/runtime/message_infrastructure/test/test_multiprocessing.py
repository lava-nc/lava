# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
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
from message_infrastructure.multiprocessing import MultiProcessing
from message_infrastructure import SendPort
from message_infrastructure import RecvPort
from message_infrastructure import Channel

import time


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


class Builder():
    def build(self, i):
        print(f"Builder running build {i}")
        print(f"Build {i}: sleep for 10s")
        time.sleep(5)
        print(f"Build {i}: Builder Achieved")


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
        print("builder", actor.get_status())
        builder.build(idx)
        return 0
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e

def multiprocessing_spawn(builder, target_fn):
        """
        Spawns an actor.
        """
        mp = MultiProcessing()
        mp.start()
        #builder = Builder()

        # Build 5 actors
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        return mp, mp.actors
class TestMultiprocessing(unittest.TestCase):
    

    def test_multiprocessing_spawn(self):
        """
        Spawns an actor.
        Checks that an actor is spawned successfully.
        """
        self.builder = Builder()

        self.mp, self.actor_list = multiprocessing_spawn(self.builder, target_fn)
        
        for actor in self.actor_list:
            print("status_run: ", actor.get_status())
            #self.assertEqual(actor.get_status(), 0)

        self.mp.stop(True)

    #@unittest.skip
    def test_multiprocessing_pause(self):
        builder = Builder()
        self.mp, self.actor_list = multiprocessing_spawn(builder, target_fn)
        
        #self.mp.pause()

        for actor in self.actor_list:
            actor.pause()
            time.sleep(0.01)
            print("status_pause: ", actor.get_status())
            #self.assertEqual(actor.get_status(), 1)

        self.mp.stop(True)

    #@unittest.skip 
    def test_multiprocessing_shutdown(self):
        """
        Spawns an actor and sends a stop signal.
        Checks that actor is stopped successfully.
        """

        self.builder = Builder()
        self.mp, self.actor_list = multiprocessing_spawn(self.builder, target_fn)

        for actor in self.actor_list:
            actor.stop()
            time.sleep(0.01)
            print("status_stop: ", actor.get_status())
            #self.assertEqual(actor.get_status(), 2)

        self.mp.stop(True)

        # actor_list = self.mp.actor_pids
        # self.mp.stop()

        # Check that all actor PIDs no longer exist
        # for actor_pid in actor_list:
        #     self.assertFalse(psutil.pid_exists(actor_pid))
   
    @unittest.skip
    def test_get_actor_list(self):
        """
        Gets list of actors
        Checks that all actors are of Actor type
        """

        self.builder = Builder()
        self.mp, self.actor_list = multiprocessing_spawn(self.builder, target_fn)
        
        for actor in self.actor_list:
            self.assertIsInstance(actor, Actor)

        self.mp.stop(True)

def test_multiprocessing():
    mp = MultiProcessing()
    mp.start()
    builder = Builder()
    for i in range(5):
        bound_target_fn = partial(target_fn, idx=i)
        ret = mp.build_actor(bound_target_fn, builder)
        #print(ret)
    time.sleep(5)
    # shmm = mp.smm
    # for i in range(5):
    #     print("shared memory id: ", shmm.alloc_mem(8))

    actors = mp.actors

    actor = actors[0]
    print("actor status0: ", actor.get_status())
    
    time.sleep(5)
    print("actor status1: ", actor.get_status())

    # print("stop num: ", shmm.stop())
    # print("stop num: ", shmm.stop())

    mp.stop(True)


# Run unit tests
if __name__ == '__main__':
    #test_multiprocessing()
    unittest.main()
