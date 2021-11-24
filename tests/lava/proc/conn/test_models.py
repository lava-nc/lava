# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF


class DenseRunConfig(RunConfig):
    """Run configuration selects appropriate Dense ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError("No legal ProcessModel found.")


class VecSendProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        send_at_times = kwargs.pop("send_at_times")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.send_at_times = Var(shape=(num_steps,), init=send_at_times)
        self.s_out = OutPort(shape=shape)
        self.a_in = InPort(shape=shape) #enables recurrence test


class VecRecvProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        self.shape = shape
        self.s_in = InPort(shape=(shape[1],))
        self.spk_data = Var(shape=shape, init=0)  # This Var expands with time


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('floating_pt')
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE,float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.current_ts - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('fixed_pt')
class PyVecSendModelFixed(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,precision=16)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.current_ts - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('floating_pt')
class PySpkRecvModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.current_ts - 1, :] = spk_in


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('fixed_pt')
class PySpkRecvModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.current_ts - 1, :] = spk_in


class TestDenseProcessModelsFloat(unittest.TestCase):
    """Tests for floating point ProcessModels of Dense"""
    def test_float_pm_buffer(self):
        """
        Tests floating point Dense ProcessModel in which all input neurons
        fire once, but only 1 connection weight is non-zero.
        """
        shape = (3,4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],),dtype=np.float)
        send_at_times = np.repeat(False,(num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[2,2] = 1
        dense = Dense(shape=shape,
                  weights=weights
            )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        #dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        #Expected behavior is that a_out corresponding to layer 1, neuron 2 will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        expected_spk_data[4, 2] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_in(self):
        """
        Tests floating point Dense ProcessModel in which all input neurons
        fire once, but only 1 connection weight is non-zero.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=np.float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[2, :] = [2,-3,4,-5]
        dense = Dense(shape=shape,
                      weights=weights
                      )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        # dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Expected behavior is that a_out corresponding to layer 1,
        # neuron 2 will be equal to 14=2-3+4-5 at timestep 5.
        expected_spk_data[4, 2] = -2
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out(self):
        """
        Tests floating point Dense ProcessModel in which all input neurons
        fire once, but only 1 connection weight is non-zero.
        """
        shape = (3,4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],),dtype=np.float)
        send_at_times = np.repeat(False,(num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[:,2] = [3,4,5]
        dense = Dense(shape=shape,
                  weights=weights
            )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        #dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        #Expected behavior is that a_out corresponding to layer 1, neuron 2 will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        expected_spk_data[4, :] = [3,4,5]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_recurrence(self):
        """
        Tests floating point Dense ProcessModel in which all input neurons
        fire once, but only 1 connection weight is non-zero.
        """
        shape = (3,3)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],),dtype=np.float)
        send_at_times = np.repeat(True,(num_steps,))
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.ones(shape, dtype=np.float)
        dense = Dense(shape=shape,
                  weights=weights
            )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        dense.stop()



class TestDenseProcessModelsFixed(unittest.TestCase):
    """Tests for fixed point, ProcessModels of Dense, which are bit-accurate
    with Loihi hardware"""
    def test_bitacc_pm_fan_out_e(self):
        """
            Tests floating point Dense ProcessModel in which all input neurons
            fire once, but only 1 connection weight is non-zero.
            """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=np.float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[:, 2] = [0.5,300,40]
        dense = Dense(shape=shape,
                      weights=weights,
                      sign_mode=2
                      )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        # dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Expected behavior is that a_out corresponding to layer 1, neuron 2
        # will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        expected_spk_data[4, :] = [0, 255, 40]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_mixed_sign(self):
        """
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254.
            """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=np.float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[:, 2] = [300,-300,39]
        dense = Dense(shape=shape,
                      weights=weights,
                      sign_mode=1
                      )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        # dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Expected behavior is that a_out corresponding to layer 1, neuron 2
        # will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        #only even values can be returned
        expected_spk_data[4, :] = [254, -256, 38]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_exp(self):
        """
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254.
            """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=np.float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[:, 2] = [300,-300,39]
        dense = Dense(shape=shape,
                      weights=weights,
                      weight_exp=1
                      )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        # dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        #print(spk_data_through_run)
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Expected behavior is that a_out corresponding to layer 1, neuron 2
        # will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        #only even values can be returned
        expected_spk_data[4, :] = [508,-512,76]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_in_mixed_sign(self):
        """
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254.
            """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=np.float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendProcess(shape=(shape[1],), num_steps=num_steps,
                             vec_to_send=vec_to_send,
                             send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight
        weights = np.zeros(shape, dtype=np.float)
        weights[2, :] = [300, -300, 39, -0.4]
        dense = Dense(shape=shape,
                      weights=weights,
                      sign_mode=1
                      )
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        # dense.compile(run_cfg=rcfg)
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        #print(spk_data_through_run)
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Expected behavior is that a_out corresponding to layer 1, neuron 2
        # will
        # be equal to 1 at timestep 5, because the dendritic accumulators work
        #  on inputs from the previous timestep.
        # only even values can be returned
        expected_spk_data[4, 2] = 36
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))



