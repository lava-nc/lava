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
from lava.magma.compiler.channels.pypychannel import (PyPyChannel, CspRecvPort,
                                                      CspSendPort)

from lava.proc.io.bridge.injector import Injector, PyLoihiInjectorModel
from lava.proc.io.bridge.utils import ChannelConfig, ChannelSendBufferFull, \
    ChannelRecvBufferEmpty, ChannelRecvBufferNotEmpty


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
                 shape: tuple[int, ...],
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

        self.assertIsInstance(injector._injector_channel_config,
                              ChannelConfig)
        self.assertEqual(
            injector._injector_channel_config.send_buffer_full,
            ChannelSendBufferFull.BLOCKING)
        self.assertEqual(
            injector._injector_channel_config.recv_buffer_empty,
            ChannelRecvBufferEmpty.BLOCKING)
        self.assertEqual(
            injector._injector_channel_config.recv_buffer_not_empty,
            ChannelRecvBufferNotEmpty.FIFO)

        self.assertIsInstance(injector._multi_processing, MultiProcessing)
        self.assertIsInstance(injector._injector_channel, PyPyChannel)
        self.assertIsInstance(injector._injector_channel_src_port, CspSendPort)

        self.assertEqual(injector.proc_params["shape"], out_shape)
        self.assertIsInstance(injector.proc_params["injector_channel_config"],
                              ChannelConfig)
        self.assertEqual(
            injector.proc_params["injector_channel_config"].send_buffer_full,
            ChannelSendBufferFull.BLOCKING)
        self.assertEqual(
            injector.proc_params["injector_channel_config"].recv_buffer_empty,
            ChannelRecvBufferEmpty.BLOCKING)
        self.assertEqual(
            injector.proc_params[
                "injector_channel_config"].recv_buffer_not_empty,
            ChannelRecvBufferNotEmpty.FIFO)
        self.assertIsInstance(injector.proc_params["injector_channel_dst_port"],
                              CspRecvPort)

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
        injector_channel_config parameter raises errors."""
        out_shape = (1,)

        channel_config = "config"
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, injector_channel_config=channel_config)

        channel_config = ChannelConfig(
            send_buffer_full=1,
            recv_buffer_empty=ChannelRecvBufferEmpty.BLOCKING,
            recv_buffer_not_empty=ChannelRecvBufferNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, injector_channel_config=channel_config)

        channel_config = ChannelConfig(
            send_buffer_full=ChannelSendBufferFull.BLOCKING,
            recv_buffer_empty=1,
            recv_buffer_not_empty=ChannelRecvBufferNotEmpty.FIFO)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, injector_channel_config=channel_config)

        channel_config = ChannelConfig(
            send_buffer_full=ChannelSendBufferFull.BLOCKING,
            recv_buffer_empty=ChannelRecvBufferEmpty.BLOCKING,
            recv_buffer_not_empty=1)
        with self.assertRaises(TypeError):
            Injector(shape=out_shape, injector_channel_config=channel_config)

    def test_send_data_invalid_data(self):
        """Test that calling send_data on an instance of the Injector Process
        with an invalid data parameter throws errors."""
        shape = (1,)

        injector = Injector(shape=shape)

        data = [1]
        with self.assertRaises(TypeError):
            injector.send_data(data)

        data = np.ones((2, ))
        with self.assertRaises(ValueError):
            injector.send_data(data)


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
                       "injector_channel_config": ChannelConfig(),
                       "injector_channel_dst_port": channel.dst_port}

        pm = PyLoihiInjectorModel(proc_params)

        self.assertIsInstance(pm, PyLoihiInjectorModel)
        self.assertEqual(pm._shape, shape)
        self.assertIsInstance(pm._injector_channel_config,
                              ChannelConfig)
        self.assertEqual(
            pm._injector_channel_config.send_buffer_full,
            ChannelSendBufferFull.BLOCKING)
        self.assertEqual(
            pm._injector_channel_config.recv_buffer_empty,
            ChannelRecvBufferEmpty.BLOCKING)
        self.assertEqual(
            pm._injector_channel_config.recv_buffer_not_empty,
            ChannelRecvBufferNotEmpty.FIFO)
        self.assertEqual(pm._injector_channel_dst_port, channel.dst_port)
        self.assertIsNotNone(pm._injector_channel_dst_port.thread)

    def test_send_data_send_buffer_full_blocking(self):
        """Test that calling send_data on an instance of the Injector Process
        with ChannelSendBufferFull.BLOCKING blocks."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = ChannelConfig(
            send_buffer_full=ChannelSendBufferFull.BLOCKING)

        num_steps = 1

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            injector.send_data(np.ones(data_shape))
            checkpoint_2 = time.perf_counter()
            injector.send_data(np.ones(data_shape))
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

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertGreater(time_2, 1)

    def test_send_data_send_buffer_full_non_blocking_drop(self):
        """Test that calling send_data on an instance of the Injector Process
        with ChannelSendBufferFull.NON_BLOCKING_DROP does not block."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = ChannelConfig(
            send_buffer_full=ChannelSendBufferFull.NON_BLOCKING_DROP)

        num_steps = 1

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        shared_queue = Queue(2)

        def thread_2_fn(queue: Queue) -> None:
            checkpoint_1 = time.perf_counter()
            injector.send_data(np.ones(data_shape))
            checkpoint_2 = time.perf_counter()
            with self.assertWarns(UserWarning):
                injector.send_data(np.ones(data_shape))
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

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertLess(time_2, 1)

    def test_send_data_recv_buffer_empty_blocking(self):
        """Test that running an instance of the Injector Process with
        ChannelRecvBufferEmpty.BLOCKING without calling send_data blocks."""
        data_shape = (1,)
        buffer_size = 1
        channel_config = ChannelConfig(
            recv_buffer_empty=ChannelRecvBufferEmpty.BLOCKING)

        num_steps = 1

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        injector.send_data(np.ones(data_shape))
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

        injector.send_data(np.ones(data_shape))

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True,
                                    args=[shared_queue])
        thread_2.start()

        time.sleep(2)

        injector.send_data(np.ones(data_shape))

        time.sleep(1)

        injector.stop()

        thread_2.join()

        time_1 = shared_queue.get()
        time_2 = shared_queue.get()

        self.assertFalse(thread_2.is_alive())
        self.assertLess(time_1, 1)
        self.assertGreater(time_2, 1)

    def test_send_data_recv_buffer_empty_non_blocking(self):
        """Test that running an instance of the Injector Process with
        ChannelRecvBufferEmpty.NON_BLOCKING_ZEROS without calling send_data
        does not block and that zeros are received instead."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = ChannelConfig(
            recv_buffer_empty=ChannelRecvBufferEmpty.NON_BLOCKING_ZEROS)

        num_steps = 1

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape, buffer_size=num_steps)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        injector.run(condition=run_condition, run_cfg=run_cfg)

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data,
                                np.zeros(data_shape)[np.newaxis, :])

    def test_send_data_recv_buffer_not_empty_fifo(self):
        """Test that running an instance of the Injector Process with
        ChannelRecvBufferNotEmpty.FIFO after calling send_data two times in a
        row has the effect of making the ProcessModel receive the two
        sent items one by one."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = ChannelConfig(
            recv_buffer_not_empty=ChannelRecvBufferNotEmpty.FIFO)

        num_steps = 2

        send_data = np.array([[10], [15]])

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape, buffer_size=num_steps)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        injector.send_data(send_data[0])
        injector.send_data(send_data[1])

        injector.run(condition=run_condition, run_cfg=run_cfg)

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data, send_data)

    def test_send_data_recv_buffer_not_empty_accumulate(self):
        """Test that running an instance of the Injector Process with
        ChannelRecvBufferNotEmpty.ACCUMULATE after calling send_data two times
        in a row has the effect of making the ProcessModel receive the two
        sent items, accumulated, in the first time step."""
        data_shape = (1,)
        buffer_size = 10
        channel_config = ChannelConfig(
            recv_buffer_not_empty=ChannelRecvBufferNotEmpty.ACCUMULATE)

        num_steps = 1

        send_data = np.array([[10], [15]])

        injector = Injector(shape=data_shape, buffer_size=buffer_size,
                            injector_channel_config=channel_config)
        recv = Recv(shape=data_shape, buffer_size=num_steps)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        injector.send_data(send_data[0])
        injector.send_data(send_data[1])

        injector.run(condition=run_condition, run_cfg=run_cfg)

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data,
                                np.sum(send_data, axis=0)[np.newaxis, :])

    def test_run_steps_blocking(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunSteps(blocking=True), for multiple time steps, with a
        separate thread calling send_data, runs and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_steps = 50
        num_send = num_steps

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_steps)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()

        send_data = np.random.random(size=(num_send, ) + data_shape)

        def thread_2_fn() -> None:
            for send_data_single_item in send_data:
                injector.send_data(send_data_single_item)

        thread_2 = threading.Thread(target=thread_2_fn,
                                    daemon=True)
        thread_2.start()

        injector.run(condition=run_condition, run_cfg=run_cfg)

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data, send_data)

    def test_run_steps_non_blocking(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunSteps(blocking=False), for multiple time steps, runs
        and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_steps = 50
        num_send = num_steps

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_steps)

        injector.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        send_data = np.random.random(size=(num_send,) + data_shape)

        injector.run(condition=run_condition, run_cfg=run_cfg)

        for send_data_single_item in send_data:
            injector.send_data(send_data_single_item)

        injector.wait()

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data, send_data)

    def test_run_continuous(self):
        """Test that running the a Lava network involving the Injector
        Process, with RunContinuous(), for multiple time steps, runs
        and terminates."""
        np.random.seed(0)

        data_shape = (1,)
        buffer_size = 10

        num_send = 50

        injector = Injector(shape=data_shape, buffer_size=buffer_size)
        recv = Recv(shape=data_shape, buffer_size=num_send)

        injector.out_port.connect(recv.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        send_data = np.random.random(size=(num_send,) + data_shape)

        injector.run(condition=run_condition, run_cfg=run_cfg)

        for send_data_single_item in send_data:
            injector.send_data(send_data_single_item)

        injector.pause()
        injector.wait()

        recv_var_data = recv.var.get()

        injector.stop()

        np.testing.assert_equal(recv_var_data[:num_send // 10],
                                send_data[:num_send // 10])
