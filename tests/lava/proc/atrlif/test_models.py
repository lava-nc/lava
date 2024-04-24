# Copyright (C) 2024 Intel Corporation
# Copyright (C) 2024 Jannik Luboeinski
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
from lava.proc.atrlif.process import ATRLIF


class AtrlifRunConfig(RunConfig):
    """Run configuration selects appropriate ATRLIF ProcessModel based on tag:
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


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# Following tag is needed to discover the ProcessModel using AtrlifRunConfig
@tag('floating_pt')
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, float)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# Following tag is needed to discover the ProcessModel using AtrlifRunConfig
@tag('fixed_pt')
class PyVecSendModelFixed(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16, precision=16)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=16)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# Following tag is needed to discover the ProcessModel using AtrlifRunConfig
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
# Following tag is needed to discover the ProcessModel using AtrlifRunConfig
@tag('fixed_pt')
class PySpkRecvModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


class TestATRLIFProcessModelsFloat(unittest.TestCase):
    """
    Tests for floating point ProcessModels of ATRLIF, resembling the
    existing tests for the LIF process.
    """
    def test_float_pm_no_decay(self):
        """
        Tests floating point ATRLIF ProcessModel with no current or voltage
        decay and neurons driven by internal biases.
        """
        shape = (10,)
        num_steps = 50
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps, then less often due to the refractor-
        # iness. For the test implementation below, `theta_0` has to be a
        # multiple of `bias`.
        bias = 2
        theta_0 = 4
        neur = ATRLIF(shape=shape,
                      delta_i=0.,
                      delta_v=0.,
                      delta_theta=0.,
                      delta_r=0.,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=0.,
                      bias_mant=bias * np.ones(shape, dtype=float))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) - 2 * theta_0 >=
        # theta_0`)
        t_spike_refr = 3 * theta_0 // bias - t_spike_0
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))

    def test_float_pm_impulse_delta_i(self):
        """
        Tests floating point ATRLIF ProcessModel's impulse response with no
        voltage decay and input activation at the very first time-step.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 8
        # Send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay = 0.5.
        # Set up high constant threshold, such that there are no output spikes.
        neur = ATRLIF(shape=shape,
                      delta_i=0.5,
                      delta_v=0.,
                      delta_theta=0.,
                      delta_r=0.,
                      theta_0=256.,
                      theta=256.,
                      theta_step=0.,
                      bias_mant=np.zeros(shape, dtype=float))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur_i = []
        # Run 1 timestep at a time and collect state variable i
        for _ in range(num_steps):
            neur.run(condition=rcnd, run_cfg=rcfg)
            neur_i.append(neur.i.get()[0])
        neur.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step
        expected_i_timeseries = [2. ** (7 - j) for j in range(8)]
        self.assertListEqual(expected_i_timeseries, neur_i)

    def test_float_pm_impulse_delta_v(self):
        """
        Tests floating point ATRLIF ProcessModel's impulse response with no
        current decay and input activation at the very first time-step.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 8
        # Send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay. Voltage decay = 0.5.
        # Set up high constant threshold, such that there are no output spikes.
        neur = ATRLIF(shape=shape,
                      delta_i=0.,
                      delta_v=0.5,
                      delta_theta=0.,
                      delta_r=0.,
                      theta_0=256.,
                      theta=256.,
                      theta_step=0.,
                      bias_mant=np.zeros(shape, dtype=float))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur_v = []
        # Run 1 timestep at a time and collect state variable v
        for _ in range(num_steps):
            neur.run(condition=rcnd, run_cfg=rcfg)
            neur_v.append(neur.v.get()[0])
        neur.stop()
        # Gold standard for testing: voltage decay of 0.5 should integrate
        # the voltage from 128. to 255., with steps of 64., 32., 16., etc.
        expected_v_timeseries = [128., 192., 224., 240.,
                                 248., 252., 254., 255.]
        self.assertListEqual(expected_v_timeseries, neur_v)

    def test_float_pm_instant_theta_decay(self):
        """
        Tests floating point ATRLIF ProcessModel's behavior for instant decay
        of the threshold variable in the presence of constant bias.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 20
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps, then less often due to the refractor-
        # iness. For the test implementation below, `theta_0` has to be a
        # multiple of `bias`. Following a spike, the threshold `theta` is
        # increased tremendously (by 10.), but this remains without effect
        # due to the instant decay (`delta_theta=1.`).
        bias = 2
        theta_0 = 4
        neur = ATRLIF(shape=shape,
                      delta_i=0.,
                      delta_v=0.,
                      delta_theta=1.,
                      delta_r=0.,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=10.,
                      bias_mant=bias * np.ones(shape, dtype=float))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) - 2 * theta_0 >=
        # theta_0`)
        t_spike_refr = 3 * theta_0 // bias - t_spike_0
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))

    def test_float_pm_instant_r_decay(self):
        """
        Tests floating point ATRLIF ProcessModel's behavior for instant decay
        of the refractory variable in the presence of constant bias.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 20
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps. Following a spike, the threshold `theta`
        # is automatically increased by `2 * theta`, but this remains without
        # effect due to the instant decay (`delta_r=1.`).
        bias = 8
        theta_0 = 16
        neur = ATRLIF(shape=shape,
                      delta_i=0.,
                      delta_v=0.,
                      delta_theta=0.,
                      delta_r=1.,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=0.,
                      bias_mant=bias * np.ones(shape, dtype=float))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) >= theta_0`)
        t_spike_refr = theta_0 // bias - t_spike_0 + 1
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))


class TestATRLIFProcessModelsFixed(unittest.TestCase):
    """
    Tests for fixed point ProcessModels of ATRLIF (which are bit-accurate
    with Loihi hardware), resembling the existing tests for the LIF process.
    """
    def test_bitacc_pm_no_decay(self):
        """
        Tests fixed point ATRLIF ProcessModel (bit-accurate
        with Loihi hardware) with no current or voltage
        decay and neurons driven by internal biases.
        """
        shape = (10,)
        num_steps = 50
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=np.int16),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 2 * 2**6 = 128 and threshold = 8<<6
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps, then less often due to the refractor-
        # iness. For the test implementation below, `theta_0` has to be a
        # multiple of `bias`.
        bias = 4
        theta_0 = 8
        neur = ATRLIF(shape=shape,
                      delta_i=0,
                      delta_v=0,
                      delta_theta=0,
                      delta_r=0,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=0,
                      bias_mant=bias * np.ones(shape, dtype=np.int32),
                      bias_exp=6 * np.ones(shape, dtype=np.int32))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='fixed_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) - 2 * theta_0 >=
        # theta_0`)
        t_spike_refr = 3 * theta_0 // bias - t_spike_0
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))

    def test_bitacc_pm_impulse_delta_i(self):
        """
        Tests fixed point ATRLIF ProcessModel's impulse response with no
        voltage decay and input activation at the very first time-step.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 8
        # Send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay is a 12-bit
        # unsigned variable in Loihi hardware. Therefore, 2**-12 is the
        # equivalent of 1. The subtracted 1 is added by default in the
        # hardware via the `ds_offset` setting, thereby finally giving
        # `delta_i = 2048 = 0.5 * 2**12`.
        # Set up threshold high, such that there are no output spikes. By
        # default the threshold value here is left-shifted by 6.
        neur = ATRLIF(shape=shape,
                      delta_i=0.5 - (2**-12),
                      delta_v=0,
                      delta_theta=0,
                      delta_r=0,
                      theta_0=256 * np.ones(shape, dtype=np.int32),
                      theta=256 * np.ones(shape, dtype=np.int32),
                      theta_step=0,
                      bias_mant=np.zeros(shape, dtype=np.int16),
                      bias_exp=np.ones(shape, dtype=np.int16))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = AtrlifRunConfig(select_tag='fixed_pt')
        neur_i = []
        # Run 1 timestep at a time and collect state variable i
        for _ in range(num_steps):
            neur.run(condition=rcnd, run_cfg=rcfg)
            neur_i.append(neur.i.get().astype(np.int32)[0])
        neur.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step.
        expected_i_timeseries = [1 << (13 - j) for j in range(8)]
        # Gold standard for floating point equivalent of the current,
        # which would be all Loihi-bit-accurate values right shifted by 6 bits
        expected_float_i = [1 << (7 - j) for j in range(8)]
        self.assertListEqual(expected_i_timeseries, neur_i)
        self.assertListEqual(expected_float_i, np.right_shift(np.array(
            neur_i), 6).tolist())

    def test_bitacc_pm_impulse_delta_v(self):
        """
        Tests fixed point ATRLIF ProcessModel's impulse response with no
        current decay and input activation at the very first time-step.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 8
        # Send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay.
        # Set up threshold high, such that there are no output spikes.
        # Threshold provided here is left-shifted by 6-bits.
        neur = ATRLIF(shape=shape,
                      delta_i=0,
                      delta_v=0.5,
                      delta_theta=0,
                      delta_r=0,
                      theta_0=256 * np.ones(shape, dtype=np.int32),
                      theta=256 * np.ones(shape, dtype=np.int32),
                      theta_step=0,
                      bias_mant=np.zeros(shape, dtype=np.int16),
                      bias_exp=np.ones(shape, dtype=np.int16))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = AtrlifRunConfig(select_tag='fixed_pt')
        neur_v = []
        # Run 1 timestep at a time and collect state variable u
        for _ in range(num_steps):
            neur.run(condition=rcnd, run_cfg=rcfg)
            neur_v.append(neur.v.get().astype(np.int32)[0])
        neur.stop()
        # Gold standard for testing: with a voltage decay of 2048, voltage
        # should integrate from 128<<6 to 255<<6. But it is slightly smaller,
        # because current decay is not exactly 0. Due to the default
        # ds_offset = 1 setting in the hardware, current decay = 1. So
        # voltage is slightly smaller than 128<<6 to 255<<6.
        expected_v_timeseries = [8192, 12286, 14331, 15351, 15859, 16111,
                                 16235, 16295]
        # Gold standard for floating point equivalent of the voltage,
        # which would be all Loihi-bit-accurate values right shifted by 6 bits
        expected_float_v = [128, 192, 224, 240, 248, 252, 254, 255]
        neur_v_float = np.right_shift(np.array(neur_v), 6)
        neur_v_float[1:] += 1  # This compensates the drift caused by ds_offset
        self.assertListEqual(expected_v_timeseries, neur_v)
        self.assertListEqual(expected_float_v, neur_v_float.tolist())

    def test_bitacc_pm_scaling_of_bias(self):
        """
        Tests fixed point ATRLIF ProcessModel's scaling of threshold.
        """
        bias_mant = 2 ** 12 - 1
        bias_exp = 5
        # Set up high threshold and high bias current to check for potential
        # overflow in effective bias in single neuron.
        neur = ATRLIF(shape=(1,),
                      delta_i=0,
                      delta_v=0.5,
                      delta_theta=0,
                      delta_r=0,
                      theta_0=2 ** 17,
                      theta=2 ** 17,
                      theta_step=0,
                      bias_mant=bias_mant,
                      bias_exp=bias_exp)

        rcnd = RunSteps(num_steps=1)
        rcfg = AtrlifRunConfig(select_tag='fixed_pt')

        neur.run(condition=rcnd, run_cfg=rcfg)
        neur_v = neur.v.get()[0]
        neur.stop()

        # Check if neur_v has correct value.
        self.assertEqual(neur_v, bias_mant * 2 ** bias_exp)

    def test_fixed_pm_instant_theta_decay(self):
        """
        Tests fixed point ATRLIF ProcessModel's behavior for instant decay
        of the threshold variable in the presence of constant bias.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 20
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps, then less often due to the refractor-
        # iness. For the test implementation below, `theta_0` has to be a
        # multiple of `bias`. Following a spike, the threshold `theta` is
        # increased tremendously (by 10.), but this remains without effect
        # due to the instant decay (`delta_theta=1.`).
        bias = 2
        theta_0 = 4
        neur = ATRLIF(shape=shape,
                      delta_i=0,
                      delta_v=0,
                      delta_theta=1,
                      delta_r=0,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=10,
                      bias_mant=bias * np.ones(shape, dtype=float))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) - 2 * theta_0 >=
        # theta_0`)
        t_spike_refr = 3 * theta_0 // bias - t_spike_0
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))

    def test_fixed_pm_instant_r_decay(self):
        """
        Tests fixed point ATRLIF ProcessModel's behavior for instant decay
        of the refractory variable in the presence of constant bias.
        """
        # Use a single neuron
        shape = (1,)
        num_steps = 20
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # `delta_i` and `delta_v` = 0 => bias driven neurons spike first after
        # `theta_0 / bias` time steps. Following a spike, the threshold `theta`
        # is automatically increased by `2 * theta`, but this remains without
        # effect due to the instant decay (`delta_r=1`).
        bias = 8
        theta_0 = 16
        neur = ATRLIF(shape=shape,
                      delta_i=0,
                      delta_v=0,
                      delta_theta=0,
                      delta_r=1,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=0,
                      bias_mant=bias * np.ones(shape, dtype=float))
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(neur.a_in)
        neur.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = AtrlifRunConfig(select_tag='floating_pt')
        neur.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        neur.stop()
        # Compute the number of time steps until the first spike
        t_spike_0 = theta_0 // bias
        # Compute the following number of time steps until the second spike
        # (according to `bias * (t_spike_0 + t_spike_refr) >= theta_0`)
        t_spike_refr = theta_0 // bias - t_spike_0 + 1
        # Gold standard for the test
        expected_spk_data = np.zeros((t_spike_0 + t_spike_refr + 1, shape[0]))
        expected_spk_data[t_spike_0 - 1:t_spike_0 + t_spike_refr + 1:
                          t_spike_refr, :] = 1.
        spk_data_through_run_needed = \
            spk_data_through_run[0:t_spike_0 + t_spike_refr + 1, :]
        self.assertTrue(np.all(expected_spk_data
                               == spk_data_through_run_needed))
