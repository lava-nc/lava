# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
from multiprocessing import shared_memory
import unittest
import numpy as np
import time


from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps


# A minimal process with an OutPort
class P1(AbstractProcess):
    def __init__(self):
        super().__init__(log_config=LogConfig(level=logging.CRITICAL))
        self.out = OutPort(shape=(2,))


# A minimal process with an InPort
class P2(AbstractProcess):
    def __init__(self):
        super().__init__(log_config=LogConfig(level=logging.CRITICAL))
        self.inp = InPort(shape=(2,))


# A minimal process with an InPort
class P3(AbstractProcess):
    def __init__(self):
        super().__init__(log_config=LogConfig(level=logging.CRITICAL))
        self.inp = InPort(shape=(2,))


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel1(PyLoihiProcessModel):
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        if self.time_step > 1:
            shm = shared_memory.SharedMemory(name='error_block')
            err = np.ndarray((1,), buffer=shm.buf)
            err[0] += 1
            shm.close()


# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel2(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_spk(self):
        if self.time_step > 1:
            # Raise exception
            # raise TypeError("All the error info")
            shm = shared_memory.SharedMemory(name='error_block')
            err = np.ndarray((1,), buffer=shm.buf)
            err[0] += 1
            shm.close()


# A minimal PyProcModel implementing P3
@implements(proc=P3, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel3(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_spk(self):
        ...


class TestExceptionHandling(unittest.TestCase):
    def setUp(self):
        """
        Creates a shared memory block.
        Runs as part of unit test method.
        """
        time.sleep(1)
        error_message = np.zeros(shape=(1,))
        shm = shared_memory.SharedMemory(create=True,
                                         size=error_message.nbytes,
                                         name='error_block')
        err = np.ndarray(error_message.shape, dtype=np.float64, buffer=shm.buf)
        err[:] = error_message[:]
        shm.close()
        self.shm_name = shm.name

    def tearDown(self):
        """
        Destroys the shared memory block.
        Runs as part of unit test method.
        """
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        existing_shm.unlink()

    def test_one_pm(self):
        """Checks the forwarding of exceptions within a ProcessModel to the
        runtime."""
        # Create an instance of P1
        proc = P1()

        run_steps = RunSteps(num_steps=1)
        run_cfg = Loihi1SimCfg(
            loglevel=logging.CRITICAL)

        # Run the network for 1 time step -> no exception
        proc.run(condition=run_steps, run_cfg=run_cfg)

        # Run the network for another time step -> expect exception
        proc.run(condition=run_steps, run_cfg=run_cfg)

        # Check that the error count has increased to 1
        existing_shm = shared_memory.SharedMemory(name='error_block')
        res = np.ndarray((1,), buffer=existing_shm.buf)
        self.assertEqual(res[0], 1)
        existing_shm.close()

    def test_two_pm(self):
        """Checks the forwarding of exceptions within two ProcessModel to the
        runtime."""
        # Create a sender instance of P1 and a receiver instance of P2
        sender = P1()
        recv = P2()

        run_steps = RunSteps(num_steps=1)
        run_cfg = Loihi1SimCfg(
            loglevel=logging.CRITICAL)

        # Connect sender with receiver
        sender.out.connect(recv.inp)

        # Run the network for 1 time step -> no exception
        sender.run(condition=run_steps, run_cfg=run_cfg)

        # Run the network for another time step -> expect exception
        sender.run(condition=run_steps, run_cfg=run_cfg)

        # Check that the error count has increased to 2
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        res = np.ndarray((1,), buffer=existing_shm.buf)
        self.assertEqual(res[0], 2)
        existing_shm.close()

    def test_three_pm(self):
        """Checks the forwarding of exceptions within three ProcessModel to the
        runtime."""
        # Create a sender instance of P1 and receiver instances of P2 and P3
        sender = P1()
        recv1 = P2()
        recv2 = P3()

        run_steps = RunSteps(num_steps=1)
        run_cfg = Loihi1SimCfg(
            loglevel=logging.CRITICAL)

        # Connect sender with receiver
        sender.out.connect([recv1.inp, recv2.inp])

        # Run the network for 1 time step -> no exception
        sender.run(condition=run_steps, run_cfg=run_cfg)

        # Run the network for another time step -> expect exception
        sender.run(condition=run_steps, run_cfg=run_cfg)

        # Check that the error count has increased to 2
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        res = np.ndarray((1,), buffer=existing_shm.buf)
        self.assertEqual(res[0], 2)
        existing_shm.close()


if __name__ == '__main__':
    unittest.main(buffer=False)
