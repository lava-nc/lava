# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
import threading
import time
import typing as ty
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

        config = extractor.proc_params["channel_config"]
        self.assertIsInstance(config, utils.ChannelConfig)
        self.assertEqual(config.send_full, utils.SendFull.BLOCKING)
        self.assertEqual(config.receive_empty, utils.ReceiveEmpty.BLOCKING)
        self.assertEqual(config.receive_not_empty, utils.ReceiveNotEmpty.FIFO)

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

    @staticmethod
    def _test_send_full_policy(send_full: utils.SendFull) \
            -> ty.Tuple[int, int, bool]:
        """Sets up a simple network involving a Send Process and an Extractor
        with buffer_size=1.

        In the main thread, the network is ran a single time (so that the
        Runtime gets created), the element present in the channel gets
        consumed, then we sleep for 2 seconds.
        In a separate thread, the network is ran (and timing is recorded),
        triggering the Send Process to send an item to Extractor, two times
        in a row.
        Finally, the main thread calls receive() on the Extractor so that the
        element present in the channel is consumed.

        The first run() will make the Extractor fill the buffer and be fast.
        The second send() will make the Extractor trigger the behavior
        determined by the SendFull policy passed as argument
        (either BLOCKING or NON_BLOCKING_DROP).

        Parameters
        ----------
        send_full : SendFull
            Enum instance specifying the SendFull policy of the channel.
        Returns
        ----------
        time_send_1 : int
            Time it took to run the first run().
        time_send_2 : int
            Time it took to run the second run().
        thread_is_alive : bool
            Boolean representing whether or not the separate thread terminated.
        """
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(send_full=send_full)
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

        time_run_1 = shared_queue.get()
        time_run_2 = shared_queue.get()

        return time_run_1, time_run_2, thread_2.is_alive()

    def test_receive_data_send_full_blocking(self):
        """Test that running an instance of the Extractor Process with
        SendFull.BLOCKING when the channel is full blocks."""
        send_full = utils.SendFull.BLOCKING

        time_run_1, time_run_2, thread_is_alive = \
            self._test_send_full_policy(send_full)

        self.assertLess(time_run_1, 1)
        self.assertGreater(time_run_2, 1)
        self.assertFalse(thread_is_alive)

    def test_receive_data_send_full_non_blocking_drop(self):
        """Test that running an instance of the Extractor Process with
        SendFull.NON_BLOCKING_DROP when the channel is full does not block."""
        send_full = utils.SendFull.NON_BLOCKING_DROP

        time_run_1, time_run_2, thread_is_alive = \
            self._test_send_full_policy(send_full)

        self.assertLess(time_run_1, 1)
        self.assertLess(time_run_2, 1)
        self.assertFalse(thread_is_alive)

    def test_receive_data_receive_empty_blocking(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveEmpty.BLOCKING blocks when the channel is empty.

        Sets up a simple network involving a Send Process and an Extractor with
        (buffer_size=1, receive_empty=ReceiveEmpty.BLOCKING).

        In the main thread, the network is ran a single time (so that the
        Extractor channel is not empty), then we sleep for 2 seconds.
        In a separate thread, we call receive() on the Extractor (and timing
        is recorded). two times in a row.
        Finally, the main thread runs the network a single time step, sleeps
        a second, and stops the network.

        The first receive() should be fast because an item is already in the
        channel.
        The second receive() should be slow because no item is in the channel
        (main thread is sleeping for 2 seconds).
        """
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
        empty and that zeros are returned instead.

        Instantiates an Extractor with
        (buffer_size=1, receive_empty=ReceiveEmpty.NON_BLOCKING_ZEROS)
        without running it.

        Checks that receive() returns zeros.
        """
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.NON_BLOCKING_ZEROS)

        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)

        recv_data = extractor.receive()

        np.testing.assert_equal(recv_data, np.zeros(data_shape))

    @staticmethod
    def _test_receive_not_empty_policy(receive_not_empty: utils.ReceiveNotEmpty,
                                       send_data: np.ndarray) -> np.ndarray:
        """Sets up a simple network involving a Send Process and an Extractor
        with buffer_size=10.

        Runs the network for 2 time steps, making the Send Process send two
        data items in a row.
        Then, call receive() on the Extractor 2 times in a row.

        Depending on the ReceiveNotEmpty policy passed as argument (either
        FIFO or ACCUMULATE), the two consecutive calls to receive() on the
        Extractor will either return the two sent items one after the other
        or the sum of the two items followed by a 0.

        Parameters
        ----------
        receive_not_empty : ReceiveNotEmpty
            Enum instance specifying the ReceiveNotEmpty policy of the channel.

        Returns
        ----------
        recv_data : np.ndarray
            Data returned by the two consecutive calls to receive() on the
            Extractor.
        """
        data_shape = (1,)
        buffer_size = 10
        # ReceiveEmpty policy is set to NON_BLOCKING_ZEROS so that the second
        # call to receive() does not block when ReceiveNotEmpty policy is set
        # to ACCUMULATE
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.NON_BLOCKING_ZEROS,
            receive_not_empty=receive_not_empty)
        num_steps = 2

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                              channel_config=channel_config)
        send.out_port.connect(extractor.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)
        recv_data = [extractor.receive(), extractor.receive()]
        extractor.stop()

        return np.array(recv_data)

    def test_receive_data_receive_not_empty_fifo(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveNotEmpty.FIFO after having sent two items in a row
        has the effect of returning the two sent items one by one."""
        receive_not_empty = utils.ReceiveNotEmpty.FIFO
        send_data = np.array([[10], [15]])

        recv_data = self._test_receive_not_empty_policy(receive_not_empty,
                                                        send_data)

        np.testing.assert_equal(recv_data, send_data)

    def test_receive_data_receive_not_empty_accumulate(self):
        """Test that calling receive on an instance of the Extractor Process
        with ReceiveNotEmpty.ACCUMULATE after having sent two items in a row
        has the effect of returning the two sent items, accumulated."""
        receive_not_empty = utils.ReceiveNotEmpty.ACCUMULATE
        send_data = np.array([[10], [15]])

        recv_data = self._test_receive_not_empty_policy(receive_not_empty,
                                                        send_data)

        np.testing.assert_equal(recv_data[0], np.sum(send_data, axis=0))

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
        num_send = 50
        send_data = np.random.random(size=(num_send,) + data_shape)

        send = Send(data=send_data)
        extractor = Extractor(shape=data_shape, buffer_size=buffer_size)
        send.out_port.connect(extractor.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        extractor.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = []
        for _ in range(num_send):
            recv_data.append(extractor.receive())

        extractor.pause()
        extractor.wait()
        extractor.stop()

        np.testing.assert_equal(recv_data, send_data)
