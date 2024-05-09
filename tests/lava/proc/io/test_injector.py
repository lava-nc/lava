# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.magma.compiler.channels.pypychannel import PyPyChannel, CspRecvPort
from lava.proc.io.injector import Injector, PyLoihiInjectorModel
from lava.proc.io import utils


class Recv(AbstractProcess):
    """Process that receives arbitrary dense data and stores it in a Var.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    buffer_size: optional, int
        Size of buffer storing received data.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 buffer_size: ty.Optional[int] = 1):
        super().__init__(shape=shape, buffer_size=buffer_size)

        self.var = Var(shape=(buffer_size, ) + shape, init=0)
        self.in_port = InPort(shape=shape)


@implements(proc=Recv, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvProcModel(PyLoihiProcessModel):
    """Receives dense data from PyInPort and stores it in a Var."""
    var: np.ndarray = LavaPyType(np.ndarray, float)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None :
        super().__init__(proc_params)
        self._buffer_size = proc_params["buffer_size"]

    def run_spk(self) -> None:
        self.var[
            (self.time_step - 1) % self._buffer_size] = self.in_port.recv()


class TestInjector(unittest.TestCase):
    def test_init(self):
        """Test that the Injector Process is instantiated correctly."""
        out_shape = (1,)

        injector = Injector(shape=out_shape)

        self.assertIsInstance(injector, Injector)

        config = injector.proc_params["channel_config"]
        self.assertEqual(injector.proc_params["shape"], out_shape)
        self.assertIsInstance(config, utils.ChannelConfig)
        self.assertEqual(config.send_full, utils.SendFull.BLOCKING)
        self.assertEqual(config.receive_empty, utils.ReceiveEmpty.BLOCKING)
        self.assertEqual(config.receive_not_empty, utils.ReceiveNotEmpty.FIFO)

        self.assertIsInstance(injector.out_port, OutPort)
        self.assertEqual(injector.out_port.shape, out_shape)

    def test_invalid_shape(self):
        """Test that instantiating the Injector Process with an invalid
        shape parameter raises errors."""
        out_shape = (1.5,)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape)

        out_shape = (-1,)
        with self.assertRaises(ValueError):
            Injector(shape=out_shape)

        out_shape = 4
        with self.assertRaises(TypeError):
            Injector(shape=out_shape)

    def test_invalid_buffer_size(self):
        """Test that instantiating the Injector Process with an invalid
        buffer_size parameter raises errors."""
        out_shape = (1,)

        buffer_size = 0.5
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, buffer_size=buffer_size)

        buffer_size = -5
        with self.assertRaises(ValueError):
            Injector(shape=out_shape, buffer_size=buffer_size)

    def test_invalid_channel_config(self):
        """Test that instantiating the Injector Process with an invalid
        channel_config parameter raises errors."""
        out_shape = (1,)

        channel_config = "config"
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=1,
            receive_empty=utils.ReceiveEmpty.BLOCKING,
            receive_not_empty=utils.ReceiveNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=utils.SendFull.BLOCKING,
            receive_empty=1,
            receive_not_empty=utils.ReceiveNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, channel_config=channel_config)

        channel_config = utils.ChannelConfig(
            send_full=utils.SendFull.BLOCKING,
            receive_empty=utils.ReceiveEmpty.BLOCKING,
            receive_not_empty=1)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, channel_config=channel_config)


class TestPyLoihiInjectorModel(unittest.TestCase):
    def test_init(self):
        """Test that the PyLoihiInjectorModel ProcessModel is instantiated
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

        proc_params = {"shape": shape,
                       "channel_config": utils.ChannelConfig(),
                       "p_to_pm_dst_port": channel.dst_port}

        pm = PyLoihiInjectorModel(proc_params)

        self.assertIsInstance(pm, PyLoihiInjectorModel)

    @staticmethod
    def _test_send_full_policy(send_full: utils.SendFull) \
            -> ty.Tuple[int, int, bool]:
        """Sets up a simple network involving an Injector with buffer_size=1
        and a Recv Process.

        The main thread sleeps for 2 seconds.
        In a separate thread, a data item is sent through the Injector
        channel (and timing is recorded), two times in a row.
        Finally, the main thread runs the network so that the element present
        in the channel is consumed.

        The first send() will fill the buffer and be fast.
        The second send() will trigger the behavior determined by the
        SendFull policy passed as argument
        (either BLOCKING or NON_BLOCKING_DROP).

        Parameters
        ----------
        send_full : SendFull
            Enum instance specifying the SendFull policy of the channel.

        Returns
        ----------
        time_send_1 : int
            Time it took to run the first send().
        time_send_2 : int
            Time it took to run the second send().
        thread_is_alive : bool
            Boolean representing whether or not the separate thread terminated.
        """
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(send_full=send_full)
        num_steps = 1
        data = np.ones(data_shape)

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            channel_config=channel_config)
        recv = Recv(shape=data_shape)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        ex = injector.compile(run_cfg=run_cfg)
        injector.create_runtime(run_cfg=run_cfg, executable=ex)

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            injector.send(data)
            checkpoint_2 = time.perf_counter()
            injector.send(data)
            checkpoint_3 = time.perf_counter()

            queue.put(checkpoint_2 - checkpoint_1)
            queue.put(checkpoint_3 - checkpoint_2)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()
        time.sleep(2)
        injector.run(condition=run_condition, run_cfg=run_cfg)
        injector.stop()
        thread_2.join()

        time_send_1 = shared_queue.get()
        time_send_2 = shared_queue.get()

        return time_send_1, time_send_2, thread_2.is_alive()

    def test_send_data_send_full_blocking(self):
        """Test that calling send on an instance of the Injector Process
        with SendFull.BLOCKING blocks when the channel is full."""
        send_full = utils.SendFull.BLOCKING

        time_send_1, time_send_2, thread_is_alive = \
            self._test_send_full_policy(send_full)

        self.assertLess(time_send_1, 1)
        self.assertGreater(time_send_2, 1)
        self.assertFalse(thread_is_alive)

    def test_send_data_send_full_non_blocking_drop(self):
        """Test that calling send on an instance of the Injector Process
        with SendFull.NON_BLOCKING_DROP does not block when the channel is
        full."""
        send_full = utils.SendFull.NON_BLOCKING_DROP

        time_send_1, time_send_2, thread_is_alive = \
            self._test_send_full_policy(send_full)

        self.assertLess(time_send_1, 1)
        self.assertLess(time_send_2, 1)
        self.assertFalse(thread_is_alive)

    def test_send_data_receive_empty_blocking(self):
        """Test that running an instance of the Injector Process with
        ReceiveEmpty.BLOCKING without calling send blocks.

        Sets up a simple network involving an Injector with
        (buffer_size=1, receive_empty=ReceiveEmpty.BLOCKING) and a Recv Process.

        In the main thread, a data item is sent through the Injector
        channel, the network is ran a single time (so that the Runtime gets
        created), another data item is sent through the Injector (so that the
        Injector channel is not empty), then we sleep for 2 seconds.
        In a separate thread, the network is ran (and timing is recorded),
        two times in a row.
        Finally, the main thread sends another item, sleeps a second,
        and stops the network.

        The first run() should be fast because an item is already in the
        channel.
        The second run() should be slow because no item is in the channel (main
        thread is sleeping for 2 seconds).
        """
        data_shape = (1,)
        buffer_size = 1
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.BLOCKING)
        num_steps = 1
        data = np.ones(data_shape)

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            channel_config=channel_config)
        recv = Recv(shape=data_shape)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        ex = injector.compile(run_cfg=run_cfg)
        injector.create_runtime(run_cfg=run_cfg, executable=ex)

        injector.send(np.ones(data_shape))
        injector.run(condition=run_condition, run_cfg=run_cfg)

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            injector.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_2 = time.perf_counter()
            injector.run(condition=run_condition, run_cfg=run_cfg)
            checkpoint_3 = time.perf_counter()

            queue.put(checkpoint_2 - checkpoint_1)
            queue.put(checkpoint_3 - checkpoint_2)

        injector.send(data)
        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()
        time.sleep(2)
        injector.send(data)
        time.sleep(1)
        injector.stop()
        thread_2.join()

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertGreater(time_2, 1)

    def test_send_data_receive_empty_non_blocking_zeros(self):
        """Test that running an instance of the Injector Process with
        ReceiveEmpty.NON_BLOCKING_ZEROS without calling send
        does not block and that zeros are received instead.

        Sets up a simple network involving an Injector with
        (buffer_size=1, receive_empty=ReceiveEmpty.NON_BLOCKING_ZEROS) and a
        Recv Process.

        Runs the network, and checks that the data stored in Recv is zeros.
        """
        data_shape = (1,)
        buffer_size = 10
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.NON_BLOCKING_ZEROS)
        num_steps = 1

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            channel_config=channel_config)
        recv = Recv(shape=data_shape, buffer_size=num_steps)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        injector.run(condition=run_condition, run_cfg=run_cfg)
        recv_var_data = recv.var.get()
        injector.stop()

        np.testing.assert_equal(recv_var_data[0], np.zeros(data_shape))

    @staticmethod
    def _test_receive_not_empty_policy(receive_not_empty: utils.ReceiveNotEmpty,
                                       send_data: np.ndarray) -> np.ndarray:
        """Sets up a simple network involving an Injector with buffer_size=10
        and a Recv Process.

        Sends 2 different data items in a row.
        Then, runs the network for 2 time steps.

        Depending on the ReceiveNotEmpty policy passed as argument (either
        FIFO or ACCUMULATE), the data stored in the Recv Process will either be
        the two sent items one after the other or the sum of the two items
        followed by a 0.

        Parameters
        ----------
        receive_not_empty : ReceiveNotEmpty
            Enum instance specifying the ReceiveNotEmpty policy of the channel.

        Returns
        ----------
        recv_var_data : np.ndarray
            Data stored in Recv after the run.
        """
        data_shape = (1,)
        buffer_size = 10
        # ReceiveEmpty policy is set to NON_BLOCKING_ZEROS so that the second
        # time step  does not block when ReceiveNotEmpty policy is set to
        # ACCUMULATE
        channel_config = utils.ChannelConfig(
            receive_empty=utils.ReceiveEmpty.NON_BLOCKING_ZEROS,
            receive_not_empty=receive_not_empty)
        num_steps = 2

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            channel_config=channel_config)
        recv = Recv(shape=data_shape, buffer_size=num_steps)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        ex = injector.compile(run_cfg=run_cfg)
        injector.create_runtime(run_cfg=run_cfg, executable=ex)

        injector.send(send_data[0])
        injector.send(send_data[1])
        injector.run(condition=run_condition, run_cfg=run_cfg)
        recv_var_data = recv.var.get()
        injector.stop()

        return recv_var_data

    def test_send_data_receive_not_empty_fifo(self):
        """Test that running an instance of the Injector Process with
        ReceiveNotEmpty.FIFO after calling send two times in a row has the
        effect of making the ProcessModel receive the two sent items one by
        one."""
        receive_not_empty = utils.ReceiveNotEmpty.FIFO
        data = np.array([[10], [15]])

        recv_var_data = self._test_receive_not_empty_policy(receive_not_empty,
                                                            data)

        np.testing.assert_equal(recv_var_data, data)

    def test_send_data_receive_not_empty_accumulate(self):
        """Test that running an instance of the Injector Process with
        ReceiveNotEmpty.ACCUMULATE after calling send two times
        in a row has the effect of making the ProcessModel receive the two
        sent items, accumulated, in the first time step."""
        receive_not_empty = utils.ReceiveNotEmpty.ACCUMULATE
        data = np.array([[10], [15]])

        recv_var_data = self._test_receive_not_empty_policy(receive_not_empty,
                                                            data)

        np.testing.assert_equal(recv_var_data[0], np.sum(data, axis=0))

    def test_run_steps_blocking(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunSteps(blocking=True), for multiple time steps, with a
        separate thread calling send, runs and terminates."""
        np.random.seed(0)
        data_shape = (1,)
        buffer_size = 10
        num_steps = 50
        num_send = num_steps
        data = np.random.random(size=(num_send,) + data_shape)

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_steps)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        ex = injector.compile(run_cfg=run_cfg)
        injector.create_runtime(run_cfg=run_cfg, executable=ex)

        def thread_2_fn() -> None:
            for send_data_single_item in data:
                injector.send(send_data_single_item)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True)
        thread_2.start()
        injector.run(condition=run_condition, run_cfg=run_cfg)
        recv_var_data = recv.var.get()
        injector.stop()
        thread_2.join()

        np.testing.assert_equal(recv_var_data, data)

    def test_run_steps_non_blocking(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunSteps(blocking=False), for multiple time steps,
        with the main thread calling send, runs and terminates."""
        np.random.seed(0)
        data_shape = (1,)
        buffer_size = 10
        num_steps = 50
        num_send = num_steps
        data = np.random.random(size=(num_send,) + data_shape)

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_steps)
        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        injector.run(condition=run_condition, run_cfg=run_cfg)

        for send_data_single_item in data:
            injector.send(send_data_single_item)

        injector.wait()
        recv_var_data = recv.var.get()
        injector.stop()

        np.testing.assert_equal(recv_var_data, data)

    def test_run_continuous(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunContinuous(), for multiple time steps, for multiple
        time steps, runs and terminates."""
        np.random.seed(0)
        data_shape = (1,)
        buffer_size = 10
        num_send = 50
        send_data = np.random.random(size=(num_send,) + data_shape)

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_send)
        injector.out_port.connect(recv.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        injector.run(condition=run_condition, run_cfg=run_cfg)

        for send_data_single_item in send_data:
            injector.send(send_data_single_item)

        injector.pause()
        injector.wait()
        recv_var_data = recv.var.get()
        injector.stop()

        # Given that we are using RunContinuous(). we cannot know how many
        # Loihi time steps have been run by the network, and thus,
        # how many items have reached the Recv Process by the time all data
        # has been sent through the Injector and the network has stopped.
        # We thus only check the equality between recv_var_data (data stored
        # in Recv) and send_data (data sent through the Injector) for a small
        # portion.
        np.testing.assert_equal(recv_var_data[:num_send // 10],
                                send_data[:num_send // 10])
