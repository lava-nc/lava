import numpy as np
import unittest
import threading
import time
from queue import Queue
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous

from lava.magma.compiler.compiler import Compiler
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime

from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.magma.compiler.channels.pypychannel import PyPyChannel, CspRecvPort, \
    CspSendPort
from lava.proc.io.extractor import Extractor, PyExtractorModelFloat

from lava.proc.io.injector import Injector, PyInjectorModelFloat, \
    PyInjectorModelFixed, InjectorSendMode, InjectorRecvMode
from lava.proc.io.source import RingBuffer


class TestExtractor(unittest.TestCase):
    def test_init(self):
        in_shape = (1,)
        size = 10
        dtype = int

        extractor = Extractor(shape=in_shape, dtype=dtype, size=size)

        self.assertIsInstance(extractor, Extractor)
        self.assertIsInstance(extractor._multi_processing, MultiProcessing)
        self.assertIsInstance(extractor._extractor_channel, PyPyChannel)
        self.assertIsInstance(extractor._extractor_channel_dst_port, CspRecvPort)
        self.assertEqual(extractor.proc_params["shape"], in_shape)
        self.assertIsInstance(extractor.proc_params["extractor_channel_src_port"],
                              CspSendPort)

        self.assertIsInstance(extractor.in_port, InPort)
        self.assertEqual(extractor.in_port.shape, in_shape)

    def test_invalid_shape(self):
        dtype = float
        size = 10

        in_shape = (1.5,)
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        in_shape = (-1,)
        with self.assertRaises(ValueError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        in_shape = 4
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

    def test_invalid_dtype(self):
        in_shape = (1,)
        size = 10

        dtype = 1
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        dtype = [1]
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        dtype = np.ones(in_shape)
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        dtype = "float"
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

    def test_invalid_size(self):
        in_shape = (1,)
        dtype = float

        size = 0.5
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, dtype=dtype, size=size)

        size = -5
        with self.assertRaises(ValueError):
            Extractor(shape=in_shape, dtype=dtype, size=size)


class TestPyExtractorModelFloat(unittest.TestCase):
    def test_init(self):
        shape = (1, )
        dtype = float
        size = 10

        proc_params = {"shape": shape}

        multi_processing = MultiProcessing()
        multi_processing.start()
        channel = PyPyChannel(message_infrastructure=multi_processing,
                              src_name="src",
                              dst_name="dst",
                              shape=shape,
                              dtype=dtype,
                              size=size)

        proc_params["extractor_channel_src_port"] = channel.src_port

        pm = PyExtractorModelFloat(proc_params)

        self.assertIsInstance(pm, PyExtractorModelFloat)
        self.assertEqual(pm._shape, shape)
        self.assertEqual(pm._extractor_channel_src_port, channel.src_port)
        self.assertIsNotNone(pm._extractor_channel_src_port.thread)


    def test_run_discard_data_lifo_clear_mode(self):
        data_shape = (1,)
        dtype = float
        size = 1
        num_steps = 10
        data_to_send = np.ones((1, 10))

        input = RingBuffer(data=data_to_send)
        extractor = Extractor(shape=data_shape, dtype=dtype, size=size)

        input.s_out.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        extractor.run(condition=run_condition, run_cfg=run_cfg)
        received_data = extractor.rcv_data()
        extractor.stop()
        np.testing.assert_equal(received_data, np.full(data_shape, 1))

"""
      def test_run_receive_all_data(self):
        data_shape = (1,)
        dtype = float
        size = 1
        num_steps = 10
        data_to_send = np.ones((1, 10))

        input = RingBuffer(data=data_to_send)
        extractor = Extractor(shape=data_shape, dtype=dtype, size=size)

        input.s_out.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        shared_list = [np.zeros(data_shape)]

        def receiver_thread(shared_list):
            while True:
                shared_list[0] += extractor.rcv_data()

        thread = threading.Thread(target=receiver_thread, daemon=True, args=[shared_list])
        thread.start()
        time.sleep(1)
        extractor.run(condition=run_condition, run_cfg=run_cfg)
        extractor.stop()

        np.testing.assert_equal(shared_list[0], np.full(data_shape, 10))
"""


