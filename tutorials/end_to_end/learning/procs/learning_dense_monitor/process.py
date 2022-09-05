# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, RefPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class LearningDenseMonitor(AbstractProcess):
    def __init__(self, shape, buffer):
        super().__init__(shape=shape, 
                         buffer=buffer)
        self.s_pre = Var(shape=(shape[1], buffer))
        self.s_pre_port = InPort(shape=(shape[1],))
        self.s_post = Var(shape=(shape[0], buffer))
        self.s_post_port = InPort(shape=(shape[0],))

        self.x1 = Var(shape=(shape[1], buffer))
        self.x1_port = RefPort(shape=(shape[1],))
        self.y1 = Var(shape=(shape[0], buffer))
        self.y1_port = RefPort(shape=(shape[0],))
        self.weights = Var(shape=shape + (buffer, ))
        self.weights_port = RefPort(shape=shape)

        # Additional
        # self.x2 = Var(shape=(shape[1], buffer))
        # self.x2_port = RefPort(shape=(shape[1], ))
        #
        # self.y2 = Var(shape=(shape[0], buffer))
        # self.y2_port = RefPort(shape=(shape[0],))
        # self.y3 = Var(shape=(shape[0], buffer))
        # self.y3_port = RefPort(shape=(shape[0],))
        #
        # self.tag_2 = Var(shape=shape + (buffer,))
        # self.tag_2_port = RefPort(shape=shape)
        # self.tag_1 = Var(shape=shape + (buffer,))
        # self.tag_1_port = RefPort(shape=shape)


@implements(proc=LearningDenseMonitor, protocol=LoihiProtocol)
@requires(CPU)
class LearningDenseMonitorProcessModel(PyLoihiProcessModel):
    s_pre: np.ndarray = LavaPyType(np.ndarray, bool)
    s_pre_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    s_post: np.ndarray = LavaPyType(np.ndarray, bool)
    s_post_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)

    x1: np.ndarray = LavaPyType(np.ndarray, float)
    x1_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    y1: np.ndarray = LavaPyType(np.ndarray, float)
    y1_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    weights_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)

    # Additional
    # x2: np.ndarray = LavaPyType(np.ndarray, float)
    # x2_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    #
    # y2: np.ndarray = LavaPyType(np.ndarray, float)
    # y2_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    # y3: np.ndarray = LavaPyType(np.ndarray, float)
    # y3_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    #
    # tag_2: np.ndarray = LavaPyType(np.ndarray, float)
    # tag_2_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    # tag_1: np.ndarray = LavaPyType(np.ndarray, float)
    # tag_1_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._buffer = proc_params["buffer"]

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        x1_data = self.x1_port.read()
        y1_data = self.y1_port.read()
        weights_data = self.weights_port.read()

        self.x1[..., (self.time_step - 1) % self._buffer] = x1_data
        self.y1[..., (self.time_step - 1) % self._buffer] = y1_data
        self.weights[..., (self.time_step - 1) % self._buffer] = weights_data

        # x2_data = self.x2_port.read()
        #
        # y2_data = self.y2_port.read()
        # y3_data = self.y3_port.read()
        #
        # tag_2_data = self.tag_2_port.read()
        # tag_1_data = self.tag_1_port.read()
        #
        # self.x2[..., (self.time_step - 1) % self._buffer] = x2_data
        #
        # self.y2[..., (self.time_step - 1) % self._buffer] = y2_data
        # self.y3[..., (self.time_step - 1) % self._buffer] = y3_data
        #
        # self.tag_2[..., (self.time_step - 1) % self._buffer] = tag_2_data
        # self.tag_1[..., (self.time_step - 1) % self._buffer] = tag_1_data

    def run_spk(self):
        s_pre_data = self.s_pre_port.recv()
        s_post_data = self.s_post_port.recv()

        self.s_pre[..., (self.time_step - 1) % self._buffer] = s_pre_data
        self.s_post[..., (self.time_step - 1) % self._buffer] = s_post_data
        