# Copyright (C) 2021-22 Intel Corporation
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
from lava.proc.dense.process import Dense, DelayDense
from lava.utils.weightutils import SignMode
from lava.proc.dense.models import AbstractPyDelayDenseModel


class DenseRunConfig(RunConfig):
    """Run configuration selects appropriate Dense ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed-point precision"""

    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError("No legal ProcessModel found.")


class VecSendandRecvProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Process also listens for incoming connections via InPort a_in. This
    allows the test Process to validate that network behavior won't deadlock
    in the presence of recurrent connections.

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        send_at_times = kwargs.pop("send_at_times")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.send_at_times = Var(shape=(num_steps,), init=send_at_times)
        self.s_out = OutPort(shape=shape)
        self.a_in = InPort(shape=shape)  # enables recurrence test


class VecRecvProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.shape = shape
        self.s_in = InPort(shape=(shape[1],))
        self.spk_data = Var(shape=shape, init=0)  # This Var expands with time


@implements(proc=VecSendandRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
@tag('floating_pt')
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecSendandRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
@tag('fixed_pt')
class PyVecSendModelFixed(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        self.a_in.recv()

        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
@tag('floating_pt')
class PySpkRecvModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using DenseRunConfig
@tag('fixed_pt')
class PySpkRecvModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


class TestDenseProcessModelFloat(unittest.TestCase):
    """Tests for floating point ProcessModels of Dense"""

    def test_float_pm_buffer(self):
        """
        Tests floating point Dense ProcessModel connectivity and temporal
                   dynamics. All input 'neurons' from the VecSendandRcv fire
                   once at time           t=4, and only 1 connection weight
                   in the Dense Process is non-zero. The          non-zero
                   connection should have an activation of 1 at timestep t=5.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight at
        # entry [2,2] of the connectivity mat.
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        dense = Dense(weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # a_out will be equal to 1 at timestep 5, because the dendritic
        #  accumulators work on inputs from the previous timestep.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 2] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_in(self):
        """
        Tests floating point Dense ProcessModel dendritic accumulation
        behavior when the fan-in to a receiving neuron is greater than 1.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Dense Process where all input layer neurons project to a
        # single output layer neuron.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [2, -3, 4, -5]
        dense = Dense(weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to -2=2-3+4-5 at timestep 5.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 2] = -2
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out(self):
        """
        Tests floating point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Dense Process where a single input layer neuron projects to
        # all output layer neurons.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        dense = Dense(weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively, at timestep 5.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [3, 4, 5]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_recurrence(self):
        """
         Tests that floating Dense ProcessModel has non-blocking dynamics for
         recurrent connectivity architectures.
         """
        shape = (3, 3)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with fully connected recurrent connectivity
        # architecture
        weights = np.ones(shape, dtype=float)
        dense = Dense(weights=weights)
        # Receive neuron spikes
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        dense.stop()


class TestDenseProcessModelFixed(unittest.TestCase):
    """Tests for fixed-point, ProcessModels of Dense, which are bit-accurate
    with Loihi hardware"""

    def test_bitacc_pm_fan_out_excitatory(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and all connections are excitatory (sign_mode = 2).
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which a single input neuron projects to all
        #  output neurons.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [0.5, 300, 40]
        dense = Dense(weights=weights,
                      sign_mode=SignMode.EXCITATORY)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 0, 255, and 40, respectively,
        # at timestep 5, because a_out can only have integer values between 0
        # and 255.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [0, 255, 40]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_mixed_sign(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which a single input neuron projects to all
        # output neurons with both excitatory and inhibitory weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        dense = Dense(weights=weights, sign_mode=SignMode.MIXED)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 254, -256, and 38, respectively,
        # at timestep 5, because a_out can only have even values between -256
        # and 254.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [254, -256, 38]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_exp(self):
        """
         Tests fixed-point Dense ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and weight_exp = 1.
         When using mixed sign weights, full 8 bit weight precision,
         and weight_exp = 1, a_out can take even values from -512 to 508.
         As a result of setting weight_exp = 1, the expected a_out result is 2x
         that of the previous unit test.
         """

        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input neurons project to a single
        # output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        # Set weight_exp = 1. This affects weight scaling.
        dense = Dense(weights=weights, weight_exp=1)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 508, -512, and 76, respectively,
        # at timestep 5, because a_out can only have values between -512
        # and 508 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [508, -512, 76]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_precision(self):
        """
         Tests fixed-point Dense ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and num_weight_bits = 7.
         When using mixed sign weights and 7 bit weight precision,
         a_out can take values from -256 to 252 such that a_out % 4 = 0.
         """

        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input neurons project to a single
        # output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        # Set num_weight_bits = 7. This affects weight scaling.
        dense = Dense(weights=weights, num_weight_bits=7)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 252, -256, and 36, respectively,
        # at timestep 5, because a_out can only have values between -256
        # and 252 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [252, -256, 36]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_in_mixed_sign(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-in of a receiving neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input layer neurons project to a
        # single output layer neuron with both excitatory and inhibitory
        # weights.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [300, -300, 39, -0.4]
        dense = Dense(weights=weights, sign_mode=SignMode.MIXED)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to 36=254-256+38-0 at timestep 5, because
        # weights can only have even values between -256 and 254.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 2] = 36
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_recurrence(self):
        """
        Tests that bit accurate Dense ProcessModel has non-blocking dynamics for
        recurrent connectivity architectures.
        """
        shape = (3, 3)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with fully connected recurrent connectivity
        # architecture.
        weights = np.ones(shape, dtype=float)
        dense = Dense(weights=weights)
        # Receive neuron spikes
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        dense.stop()


class TestDelayDenseProcessModel(unittest.TestCase):
    """Tests for ProcessModels of Dense with synaptic delay."""

    def test_matrix_weight_delay_expansion(self):
        """"""
        shape = (3, 4)
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        delays = np.zeros(shape, dtype=int)
        delays[2, 2] = 2
        max_delay = np.max(delays) + 1  # setting it more than maximum delay
        wgt_dly = AbstractPyDelayDenseModel.get_delay_wgts_mat(weights,
                                                               delays,
                                                               max_delay)
        # Expected shape is maximum delay=2 + 1 = 3 times first dimension of
        # original shape (3, 4) => (9, 4)
        expected_shape = ((max_delay + 1) * 3, 4)
        self.assertTrue(np.shape(wgt_dly) == expected_shape)
        # Expected matrix stacks n zero matrices of original shape of the
        # weights matrix vertically to create the wgt_dly matrix. n being the
        # maximum delay in the delay matrix.
        expected_wgt_dly = np.zeros(expected_shape)
        expected_wgt_dly[8, 2] = 1
        self.assertTrue(np.all(wgt_dly == expected_wgt_dly))

    def test_float_pm_buffer_delay(self):
        """Tests floating point Dense ProcessModel connectivity and temporal
        dynamics. All input 'neurons' from the VecSendandRcv fire
        once at time t=4, and only 1 connection weight
        in the Dense Process is non-zero. The value of the delay matrix for
        this weight is 2. The non-zero connection should have an activation of
        1 at timestep t=7.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with a single non-zero connection weight at
        # entry [2, 2] of the connectivity matrix and a delay of 2 at entry
        # [2, 2] in the delay matrix.
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        delays = np.zeros(shape, dtype=int)
        delays[2, 2] = 2
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # a_out will be equal to 1 at timestep 7, because the dendritic
        #  accumulators work on inputs from the previous timestep + 2.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, 2] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_in_delay(self):
        """
        Tests floating point Dense ProcessModel dendritic accumulation
        behavior when the fan-in to a receiving neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 10
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Dense Process where all input layer neurons project to a
        # single output layer neuron with varying delays.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [2, -3, 4, -5]
        delays = np.zeros(shape, dtype=int)
        delays[2, :] = [1, 2, 2, 4]
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to 2 at timestep 6, 1=-3+4 at timestep 7 and
        # -5 at timestep 9
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[5, 2] = 2
        expected_spk_data[6, 2] = 1
        expected_spk_data[8, 2] = -5
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out_delay(self):
        """
        Tests floating point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Dense Process where a single input layer neuron projects to
        # all output layer neurons with a delay of 2 for all synapses.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively, at timestep 7.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [3, 4, 5]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out_delay_2(self):
        """
        Tests floating point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and synaptic delays are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up a Dense Process where a single input layer neuron projects to
        # all output layer neurons with varying delays.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively, at timestep
        # 5, 6 and 7, respectively.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 3
        expected_spk_data[5, 1] = 4
        expected_spk_data[6, 2] = 5
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_recurrence_delays(self):
        """
         Tests that floating Dense ProcessModel has non-blocking dynamics for
         recurrent connectivity architectures and synaptic delays are
         configured.
         """
        shape = (3, 3)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with fully connected recurrent connectivity
        # architecture
        weights = np.ones(shape, dtype=float)
        delays = np.zeros(shape, dtype=int)
        delays = 2
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='floating_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        dense.stop()

    def test_bitacc_pm_fan_out_excitatory_delay(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and all connections are excitatory (sign_mode = 2) and synaptic delays
        are configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which a single input neuron projects to all
        #  output neurons.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [0.5, 300, 40]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        dense = DelayDense(weights=weights, delays=delays,
                           sign_mode=SignMode.EXCITATORY)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 0, 255, and 40, respectively,
        # at timestep 5, 6 and 7, because a_out can only have integer values
        # between 0 and 255 and we have a delay of 0, 1, 2 on the synapses,
        # respectively.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 0
        expected_spk_data[5, 1] = 255
        expected_spk_data[6, 2] = 40
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_mixed_sign_delay(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254. A delay of 2 for all
        synapses is configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which a single input neuron projects to all
        # output neurons with both excitatory and inhibitory weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        dense = DelayDense(weights=weights, delays=delays,
                           sign_mode=SignMode.MIXED)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 254, -256, and 38, respectively,
        # at timestep 7, because a_out can only have even values between -256
        # and 254 and a delay of 2 is configured.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [254, -256, 38]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_exp_delay(self):
        """
         Tests fixed-point Dense ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and weight_exp = 1.
         When using mixed sign weights, full 8 bit weight precision,
         and weight_exp = 1, a_out can take even values from -512 to 508.
         As a result of setting weight_exp = 1, the expected a_out result is 2x
         that of the previous unit test. A delay of 0, 1, 2 is configured for
         respective synapses.
         """

        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input neurons project to a single
        # output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays[:, 2] = [0, 1, 2]
        # Set weight_exp = 1. This affects weight scaling.
        dense = DelayDense(weights=weights, weight_exp=1, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 508, -512, and 76, respectively,
        # at timestep 5, 6, and 7, respectively, because a_out can only
        # have values between -512 and 508 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 0] = 508
        expected_spk_data[5, 1] = -512
        expected_spk_data[6, 2] = 76
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_out_weight_precision_delay(self):
        """
         Tests fixed-point Dense ProcessModel dendritic accumulation
         behavior when the fan-out of a projecting neuron is greater than 1
         , connections are both excitatory and inhibitory (sign_mode = 1),
         and num_weight_bits = 7.
         When using mixed sign weights and 7 bit weight precision,
         a_out can take values from -256 to 252 such that a_out % 4 = 0.
         All synapses have a delay of 2 configured.
         """

        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input neurons project to a single
        # output neuron with mixed sign connection weights.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [300, -300, 39]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        # Set num_weight_bits = 7. This affects weight scaling.
        dense = DelayDense(weights=weights, num_weight_bits=7, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 252, -256, and 36, respectively,
        # at timestep 7, because a_out can only have values between -256
        # and 252 such that a_out % 4 = 0.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, :] = [252, -256, 36]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_fan_in_mixed_sign_delay(self):
        """
        Tests fixed-point Dense ProcessModel dendritic accumulation
        behavior when the fan-in of a receiving neuron is greater than 1
        and connections are both excitatory and inhibitory (sign_mode = 1).
        When using mixed sign weights and full 8 bit weight precision,
        a_out can take even values from -256 to 254. All synapses have a
        delay of 2 configured.
        """
        shape = (3, 4)
        num_steps = 8
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process in which all input layer neurons project to a
        # single output layer neuron with both excitatory and inhibitory
        # weights.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [300, -300, 39, -0.4]
        delays = np.zeros(shape, dtype=int)
        delays = 2
        dense = DelayDense(weights=weights, sign_mode=SignMode.MIXED,
                           delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        dense.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to 36=254-256+38-0 at timestep 7, because
        # weights can only have even values between -256 and 254.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[6, 2] = 36
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_recurrence_delay(self):
        """
        Tests that bit accurate Dense ProcessModel has non-blocking dynamics for
        recurrent connectivity architectures. All synapses have a delay of 2
        configured.
        """
        shape = (3, 3)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(shape=(shape[1],), num_steps=num_steps,
                                    vec_to_send=vec_to_send,
                                    send_at_times=send_at_times)
        # Set up Dense Process with fully connected recurrent connectivity
        # architecture.
        weights = np.ones(shape, dtype=float)
        delays = np.zeros(shape, dtype=int)
        delays = 2
        dense = DelayDense(weights=weights, delays=delays)
        # Receive neuron spikes
        sps.s_out.connect(dense.s_in)
        dense.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DenseRunConfig(select_tag='fixed_pt')
        dense.run(condition=rcnd, run_cfg=rcfg)
        dense.stop()
