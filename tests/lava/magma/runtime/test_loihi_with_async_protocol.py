# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.magma.core.decorator import requires, implements
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class AsyncProcessDest(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.in_port = InPort(shape=shape)


class AsyncProcessSrc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.out_port = OutPort(shape=shape)


@implements(proc=AsyncProcessDest, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessDestProcessModel(PyAsyncProcessModel):
    in_port = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_async(self):
        count = 1
        while True:
            val = self.in_port.recv()
            exp = (np.ones(
                shape=self.in_port.shape) * count).tolist()
            if val.tolist() != exp:
                raise ValueError(f"Wrong value of val : {val.tolist()} {exp}")
            count += 1
            if count == 11:
                return


@implements(proc=AsyncProcessSrc, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessSrcProcessModel(PyAsyncProcessModel):
    out_port = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_async(self):
        count = 1
        while True:
            self.out_port.send(np.ones(shape=self.out_port.shape) * count)
            count += 1
            if count == 11:
                return


class LoihiProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.var = Var(shape=shape, init=0)
        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)


@implements(proc=LoihiProcess, protocol=LoihiProtocol)
@requires(CPU)
class LoihiProcessModel(PyLoihiProcessModel):
    in_port = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port = LavaPyType(PyOutPort.VEC_DENSE, int)
    var = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        var = self.in_port.recv()
        self.out_port.send(var)


class TestProcess(unittest.TestCase):
    """This test checks if Process with Loihi Protocol works properly with
    process with Async Protocol.
    """

    def test_async_with_loihi_protocol(self):
        """
        Test is to send the data to AsyncProcessSrc from AsyncProcessSrc via
        LoihiProcess.
        AsyncProcessSrc -- > LoihiProcess --> AsyncProcessSrc
        """
        shape = (2,)
        loihi_process = LoihiProcess(shape=shape)
        async_process_dest = AsyncProcessDest(shape=shape)
        async_process_src = AsyncProcessSrc(shape=shape)
        loihi_process.in_port.connect_from(async_process_src.out_port)
        async_process_dest.in_port.connect_from(loihi_process.out_port)
        loihi_process.run(condition=RunSteps(num_steps=10),
                          run_cfg=Loihi2SimCfg())
        loihi_process.stop()
