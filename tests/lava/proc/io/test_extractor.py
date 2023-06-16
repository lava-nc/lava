# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
import threading
import time
from queue import Queue

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.magma.compiler.channels.pypychannel import PyPyChannel, CspSendPort
from lava.proc.io.extractor import Extractor, PyLoihiExtractorModel
from lava.proc.io import utils


class Send(AbstractProcess):
    """Process that sends arbitrary dense data stored in a ring buffer.

    Parameters
    ----------
    data: np.ndarray
        Data to send. Has to be at least 2D, where first dimension is time.
    """

    def __init__(self,
                 data: np.ndarray) -> None:
        super().__init__(data=data)

        self.var = Var(shape=data.shape, init=data)
        self.out_port = OutPort(shape=data.shape[1:])


@implements(proc=Send, protocol=LoihiProtocol)
@requires(CPU)
class PySendProcModel(PyLoihiProcessModel):
    """Sends dense data stored in the Var to PyOutPort."""
    var: np.ndarray = LavaPyType(np.ndarray, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        data = self.var[(self.time_step - 1) % self.var.shape[0]]
        self.out_port.send(data)


class TestExtractor(unittest.TestCase):
    def test_init(self):
        """Test that the Extractor Process is instantiated correctly."""
        in_shape = (1,)

        extractor = Extractor(shape=in_shape)

        self.assertIsInstance(extractor, Extractor)

        self.assertIsInstance(extractor.proc_params["channel_config"],
                              utils.ChannelConfig)
        self.assertEqual(extractor.proc_params["channel_config"].send_full,
                         utils.SendFull.BLOCKING)
        self.assertEqual(extractor.proc_params["channel_config"].receive_empty,
                         utils.ReceiveEmpty.BLOCKING)
        self.assertEqual(
            extractor.proc_params["channel_config"].receive_not_empty,
            utils.ReceiveNotEmpty.FIFO)
        self.assertIsInstance(extractor.proc_params["pm_to_p_src_port"],
                              CspSendPort)

        self.assertIsInstance(extractor.in_port, InPort)
        self.assertEqual(extractor.in_port.shape, in_shape)

    def test_invalid_shape(self):
        """Test that instantiating the Extractor Process with an invalid
        shape parameter raises errors."""
        in_shape = (1.5,)
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape)

        in_shape = (-1,)
        with self.assertRaises(ValueError):
            Extractor(shape=in_shape)

        in_shape = 4
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape)

    def test_invalid_buffer_size(self):
        """Test that instantiating the Extractor Process with an invalid
        buffer_size parameter raises errors."""
        in_shape = (1,)

        buffer_size = 0.5
        with self.assertRaises(TypeError):
            Extractor(shape=in_shape, buffer_size=buffer_size)

        buffer_size = -5
        with self.assertRaises(ValueError):
            Extractor(shape=in_shape, buffer_size=buffer_size)

    def test_invalid_channel_config(self):
        """Test that instantiating the Extractor Process with an invalid
        channel_config parameter raises errors."""
        out_shape = (1,)

        channel_config = "config"
        with self.assertRaises(TypeError):
            Extractor(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=1,
            receive_empty=utils.ReceiveEmpty.BLOCKING,
            receive_not_empty=utils.ReceiveNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Extractor(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=utils.SendFull.BLOCKING,
            receive_empty=1,
            receive_not_empty=utils.ReceiveNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Extractor(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=utils.SendFull.BLOCKING,
            receive_empty=utils.ReceiveEmpty.BLOCKING,
            receive_not_empty=1)
        with self.assertRaises(TypeError):
            Extractor(shape=out_shape, channel_config=channel_config)


class TestPyLoihiExtractorModel(unittest.TestCase):
    def test_init(self):
        """Test that the PyLoihiExtractorModel ProcessModel is instantiated
        correctly."""
        shape = (1, )
        buffer_size = 10

        multi_processing = MultiProcessing()
        multi_processing.start()
        channel = PyPyChannel(message_infrastructure=multi_processing,
                              src_name="src",
                              dst_name="dst",
                              shape=shape,
                              dtype=float,
                              size=buffer_size)

        proc_params = {"channel_config": utils.ChannelConfig(),
                       "pm_to_p_src_port": channel.src_port}

        pm = PyLoihiExtractorModel(proc_params)

        self.assertIsInstance(pm, PyLoihiExtractorModel)

    def test_receive_data_send_full_blocking(self):
        """Test that running an instance of the Extractor Process with
        SendFull.BLOCKING when the channel is full blocks."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(send_full=utils.SendFull.BLOCKING)

        num_steps = 1

        data = np.ones((num_steps,) + data_shape)

        send = Send(data=data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)
        extractor.receive()

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            extractor.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_2 = time.perf_counter()
            extractor.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_3 = time.perf_counter()

            queue.put(checkpoint_2 - checkpoint_1)
            queue.put(checkpoint_3 - checkpoint_2)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()

        time.sleep(2)

        extractor.receive()

        time.sleep(1)

        extractor.stop()

        thread_2.join()

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertGreater(time_2, 1)

    def test_receive_data_send_full_non_blocking_drop(self):
        """Test that running an instance of the Extractor Process with
        SendFull.NON_BLOCKING_DROP when the channel is full does not block."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(
            send_full=utils.SendFull.NON_BLOCKING_DROP)

        num_steps = 1

        data = np.ones((num_steps,) + data_shape)

        send = Send(data=data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)
        extractor.receive()

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            extractor.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_2 = time.perf_counter()
            extractor.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_3 = time.perf_counter()

            queue.put(checkpoint_2 - checkpoint_1)
            queue.put(checkpoint_3 - checkpoint_2)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()

        time.sleep(2)

        extractor.receive()

        time.sleep(1)

        extractor.stop()

        thread_2.join()

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertLess(time_2, 1)

    def test_receive_data_receive_empty_blocking(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveEmpty.BLOCKING blocks when the channel is empty."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.BLOCKING)

        num_steps = 1

        data = np.ones((num_steps,) + data_shape)

        send = Send(data=data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            extractor.receive()
            checkpoint_2 = time.perf_counter()
            extractor.receive()
            checkpoint_3 = time.perf_counter()

            queue.put(checkpoint_2 - checkpoint_1)
            queue.put(checkpoint_3 - checkpoint_2)

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()

        time.sleep(2)

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        time.sleep(1)

        extractor.stop()

        thread_2.join()

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertGreater(time_2, 1)

    def test_receive_data_receive_empty_non_blocking_zeros(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveEmpty.NON_BLOCKING_ZEROS does not block when the channel is
        empty and that zeros are returned instead."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.NON_BLOCKING_ZEROS)

        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        recv_data = extractor.receive()

        np.testing.assert_equal(recv_data,
                                np.zeros(data_shape))

    def test_receive_data_receive_not_empty_fifo(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveNotEmpty.FIFO after having sent two items in a row
        has the effect of returning the two sent items one by one."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = utils.ChannelConfig(
            receive_not_empty=utils.ReceiveNotEmpty.FIFO)

        num_steps = 2

        send_data = np.array([[10], [15]])

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = [extractor.receive(), extractor.receive()]

        extractor.stop()

        np.testing.assert_equal(recv_data, send_data)

    def test_receive_data_receive_not_empty_accumulate(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveNotEmpty.ACCUMULATE after having sent two items in a row
        has the effect of returning the two sent items, accumulated."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = utils.ChannelConfig(
            receive_not_empty=utils.ReceiveNotEmpty.ACCUMULATE)

        num_steps = 2

        send_data = np.array([[10], [15]])

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = [extractor.receive()]

        extractor.stop()

        np.testing.assert_equal(recv_data,
                                np.sum(send_data, axis=0)[np.newaxis, :])

    def test_run_steps_blocking(self):
        """Test that running the a Lava network involving the Extractor
        Process, with RunSteps(blocking=True), for multiple time steps, with a
        separate thread calling receive, runs and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_steps = 50
        num_send = num_steps

        send_data = np.random.random(size=(num_send,) + data_shape)

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        shared_queue = Queue(num_steps)

        def thread_2_fn(queue: Queue) -> None:
            for _ in range(num_steps):
                queue.put(extractor.receive())

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        extractor.stop()

        np.testing.assert_equal(list(shared_queue.queue), send_data)

    def test_run_steps_non_blocking(self):
        """Test that running the a Lava network involving the Extractor
        Process, with RunSteps(blocking=False), for multiple time steps, with
        the main thread calling receive, runs and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_steps = 50
        num_send = num_steps

        send_data = np.random.random(size=(num_send,) + data_shape)

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size)

        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = []

        for _ in range(num_steps):
            recv_data.append(extractor.receive())

        extractor.wait()

        extractor.stop()

        np.testing.assert_equal(recv_data, send_data)

    def test_run_continuous(self):
        """Test that running the a Lava network involving the Extractor
        Process, with RunContinuous(), for multiple time steps, with
        the main thread calling receive, runs and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_steps = 50
        num_send = num_steps

        send_data = np.random.random(size=(num_send,) + data_shape)

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size)

        send.out_port.connect(extractor.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = []

        for _ in range(num_steps):
            recv_data.append(extractor.receive())

        extractor.pause()
        extractor.wait()

        extractor.stop()

        np.testing.assert_equal(recv_data[:num_send // 10],
                                send_data[:num_send // 10])
