# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi2SimCfg, RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.process import LIF, LIFReset, TernaryLIF, LIFRefractory
from lava.proc import io


class LifRunConfig(RunConfig):
    """Run configuration selects appropriate LIF ProcessModel based on tag:
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
# need the following tag to discover the ProcessModel using LifRunConfig
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
# need the following tag to discover the ProcessModel using LifRunConfig
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
# need the following tag to discover the ProcessModel using LifRunConfig
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
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('fixed_pt')
class PySpkRecvModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


class TestLIFProcessModelsFloat(unittest.TestCase):
    """Tests for floating point ProcessModels of LIF"""
    def test_float_pm_no_decay(self):
        """
        Tests floating point LIF ProcessModel with no current or voltage
        decay and neurons driven by internal biases.
        """
        shape = (10,)
        num_steps = 10
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 1 * 2**1 = 2. and threshold = 4.
        # du and dv = 0 => bias driven neurons spike at every 2nd time-step.
        lif = LIF(shape=shape,
                  du=0.,
                  dv=0.,
                  bias_mant=np.ones(shape, dtype=float),
                  bias_exp=np.ones(shape, dtype=float),
                  vth=4.)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        lif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4:10:5, :] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_impulse_du(self):
        """
        Tests floating point LIF ProcessModel's impulse response with no
        voltage decay and input activation at the very first time-step.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay = 0.5
        # Set up threshold high, such that there are no output spikes
        lif = LIF(shape=shape,
                  du=0.5, dv=0,
                  bias_mant=np.zeros(shape, dtype=float),
                  bias_exp=np.ones(shape, dtype=float),
                  vth=256.)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(lif.u.get()[0])
        lif.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step
        expected_u_timeseries = [2. ** (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)

    def test_float_pm_impulse_dv(self):
        """
        Tests floating point LIF ProcessModel's impulse response with no
        current decay and input activation at the very first time-step.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay. Voltage decay = 0.5
        # Set up threshold high, such that there are no output spikes
        lif = LIF(shape=shape,
                  du=0, dv=0.5,
                  bias_mant=np.zeros(shape, dtype=float),
                  bias_exp=np.ones(shape, dtype=float),
                  vth=256.)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for _ in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(lif.v.get()[0])
        lif.stop()
        # Gold standard for testing: voltage decay of 0.5 should integrate
        # the voltage from 128. to 255., with steps of 64., 32., 16., etc.
        expected_v_timeseries = [128., 192., 224., 240., 248., 252., 254., 255.]
        self.assertListEqual(expected_v_timeseries, lif_v)


class TestLIFProcessModelsFixed(unittest.TestCase):
    """Tests for fixed point, ProcessModels of LIF, which are bit-accurate
    with Loihi hardware"""
    def test_bitacc_pm_no_decay(self):
        """
        Tests fixed point LIF ProcessModel (bit-accurate
        with Loihi hardware) with no current or voltage
        decay and neurons driven by internal biases.
        """
        shape = (10,)
        num_steps = 10
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=np.int16),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 2 * 2**6 = 128 and threshold = 8<<6
        # du and dv = 0 => bias driven neurons spike at every 4th time-step.
        lif = LIF(shape=shape,
                  du=0, dv=0,
                  bias_mant=2 * np.ones(shape, dtype=np.int32),
                  bias_exp=6 * np.ones(shape, dtype=np.int32),
                  vth=8)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        lif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4:10:5, :] = 1
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_impulse_du(self):
        """
        Tests fixed point LIF ProcessModel's impulse response with no
        voltage decay and input activation at the very first time-step.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay is a 12-bit
        # unsigned variable in Loihi hardware. Therefore, du = 2047 is
        # equivalent to (1/2) * (2**12) - 1. The subtracted 1 is added by
        # default in the hardware, via a setting ds_offset, thereby finally
        # giving du = 2048 = 0.5 * 2**12
        # Set up threshold high, such that there are no output spikes. By
        # default the threshold value here is left-shifted by 6.
        lif = LIF(shape=shape,
                  du=2047, dv=0,
                  bias_mant=np.zeros(shape, dtype=np.int16),
                  bias_exp=np.ones(shape, dtype=np.int16),
                  vth=256 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(lif.u.get().astype(np.int32)[0])
        lif.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step.
        expected_u_timeseries = [1 << (13 - j) for j in range(8)]
        # Gold standard for floating point equivalent of the current,
        # which would be all Loihi-bit-accurate values right shifted by 6 bits
        expected_float_u = [1 << (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)
        self.assertListEqual(expected_float_u, np.right_shift(np.array(
            lif_u), 6).tolist())

    def test_bitacc_pm_impulse_dv(self):
        """
        Tests fixed point LIF ProcessModel's impulse response with no
        current decay and input activation at the very first time-step.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay. Voltage decay is a 12-bit
        # unsigned variable in Loihi hardware. Therefore, dv = 2048 is
        # equivalent to (1/2) * (2**12).
        # Set up threshold high, such that there are no output spikes.
        # Threshold provided here is left-shifted by 6-bits.
        lif = LIF(shape=shape,
                  du=0, dv=2048,
                  bias_mant=np.zeros(shape, dtype=np.int16),
                  bias_exp=np.ones(shape, dtype=np.int16),
                  vth=256 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(lif.v.get().astype(np.int32)[0])
        lif.stop()
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
        lif_v_float = np.right_shift(np.array(lif_v), 6)
        lif_v_float[1:] += 1  # This compensates the drift caused by dsOffset
        self.assertListEqual(expected_v_timeseries, lif_v)
        self.assertListEqual(expected_float_v, lif_v_float.tolist())

    def test_bitacc_pm_scaling_of_bias(self):
        """
        Tests fixed point LIF ProcessModel's scaling of threshold.
        """
        bias_mant = 2 ** 12 - 1
        bias_exp = 5
        # Set up high threshold and high bias current to check for potential
        # overflow in effective bias in single neuron.
        lif = LIF(shape=(1,),
                  du=0,
                  dv=0,
                  bias_mant=bias_mant,
                  bias_exp=bias_exp,
                  vth=2 ** 17)

        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')

        lif.run(condition=rcnd, run_cfg=rcfg)
        lif_v = lif.v.get()[0]
        lif.stop()

        # Check if lif_v has correct value.
        self.assertEqual(lif_v, bias_mant * 2 ** bias_exp)


class TestTLIFProcessModelsFloat(unittest.TestCase):
    """Tests for ternary LIF floating point neuron model"""
    def test_float_pm_neg_no_decay_1(self):
        """Tests floating point ternary LIF model with negative bias
        driving a neuron without any decay of current and voltage states."""
        shape = (10,)
        num_steps = 30
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 1 * 2**1 = 2. and threshold = 4.
        # du and dv = 0 => bias driven neurons spike at every 2nd time-step.
        tlif = TernaryLIF(shape=shape, du=0., dv=0.,
                          bias_mant=(-1) * np.ones(shape, dtype=float),
                          bias_exp=np.ones(shape, dtype=float),
                          vth_lo=-7., vth_hi=5.)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        tlif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        tlif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[7:30:8, :] = -1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_neg_no_decay_2(self):
        """Tests +1 and -1 spike responses of a floating point ternary LIF
        model driven by alternating spiking inputs. No current or voltage
        decay, no bias."""
        shape = (10,)
        num_steps = 11
        pos_idx = np.hstack((np.arange(3), np.arange(9, 11)))
        send_steps_pos = np.zeros((num_steps,), dtype=bool)
        send_steps_pos[pos_idx] = True
        send_steps_neg = (1 - send_steps_pos).astype(bool)
        # Set up external input to 0
        sps_pos = VecSendProcess(shape=shape, num_steps=num_steps,
                                 vec_to_send=np.ones(shape, dtype=float),
                                 send_at_times=send_steps_pos)
        sps_neg = VecSendProcess(shape=shape, num_steps=num_steps,
                                 vec_to_send=(-1) * np.ones(shape, dtype=float),
                                 send_at_times=send_steps_neg)
        # Set up bias = 1 * 2**1 = 2. and threshold = 4.
        # du and dv = 0 => bias driven neurons spike at every 2nd time-step.
        tlif = TernaryLIF(shape=shape, du=0., dv=0.,
                          bias_mant=np.zeros(shape, dtype=float),
                          bias_exp=np.ones(shape, dtype=float),
                          vth_lo=-3., vth_hi=5.)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps_pos.s_out.connect(tlif.a_in)
        sps_neg.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        tlif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        tlif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[2, :] = 1.
        expected_spk_data[9, :] = -1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_neg_impulse_du(self):
        """Tests the impulse response of the floating point ternary LIF
        neuron model with current decay but without voltage decay"""
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of -128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(-2 ** 7) * np.ones(shape,
                                                             dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay = 0.5
        # Set up threshold high, such that there are no output spikes
        tlif = TernaryLIF(shape=shape,
                          du=0.5, dv=0,
                          bias_mant=np.zeros(shape, dtype=float),
                          bias_exp=np.ones(shape, dtype=float),
                          vth_lo=-256., vth_hi=2)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            tlif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(tlif.u.get()[0])
        tlif.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step
        expected_u_timeseries = [-2. ** (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)

    def test_float_pm_neg_impulse_dv(self):
        """Tests the impulse response of the floating point ternary LIF
        neuron model with voltage decay but without current decay"""
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of -128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(-2 ** 7) * np.ones(shape,
                                                             dtype=float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay. Voltage decay = 0.5
        # Set up threshold high, such that there are no output spikes
        tlif = TernaryLIF(shape=shape,
                          du=0, dv=0.5,
                          bias_mant=np.zeros(shape, dtype=float),
                          bias_exp=np.ones(shape, dtype=float),
                          vth_lo=-256., vth_hi=2.)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            tlif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(tlif.v.get()[0])
        tlif.stop()
        # Gold standard for testing: voltage decay of 0.5 should integrate
        # the voltage from -128. to -255., with steps of -64., -32., -16., etc.
        expected_v_timeseries = [-128., -192., -224., -240., -248., -252.,
                                 -254., -255.]
        self.assertListEqual(expected_v_timeseries, lif_v)


class TestTLIFProcessModelsFixed(unittest.TestCase):
    """Tests for ternary LIF fixed point neuron model"""
    def test_fixed_pm_neg_no_decay_1(self):
        """Tests fixed point ProcessModel for ternary LIF neurons without any
        current or voltage decay, solely driven by (negative) bias"""
        shape = (5,)
        num_steps = 10
        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=np.int16),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 2 * 2**6 = 128 and threshold = 8<<6
        # du and dv = 0 => bias driven neurons spike at every 4th time-step.
        tlif = TernaryLIF(shape=shape,
                          du=0, dv=0,
                          bias_mant=(-2) * np.ones(shape, dtype=np.int32),
                          bias_exp=6 * np.ones(shape, dtype=np.int32),
                          vth_lo=(-8), vth_hi=2)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        tlif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        tlif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4:10:5, :] = -1
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_fixed_pm_neg_no_decay_2(self):
        """Tests fixed point ProcessModel for ternary LIF neurons without any
        current or voltage decay, driven by positive and negative spikes and
        no bias."""
        shape = (10,)
        num_steps = 11
        pos_idx = np.hstack((np.arange(3), np.arange(9, 11)))
        send_steps_pos = np.zeros((num_steps,), dtype=bool)
        send_steps_pos[pos_idx] = True
        send_steps_neg = (1 - send_steps_pos).astype(bool)
        # Set up external input to 0
        sps_pos = VecSendProcess(shape=shape, num_steps=num_steps,
                                 vec_to_send=np.ones(shape, dtype=np.int32),
                                 send_at_times=send_steps_pos)
        sps_neg = VecSendProcess(shape=shape, num_steps=num_steps,
                                 vec_to_send=(-1) * np.ones(shape,
                                                            dtype=np.int32),
                                 send_at_times=send_steps_neg)
        # Set up bias = 1 * 2**1 = 2. and threshold = 4.
        # du and dv = 0 => bias driven neurons spike at every 2nd time-step.
        tlif = TernaryLIF(shape=shape, du=0, dv=0,
                          bias_mant=np.zeros(shape, dtype=np.int32),
                          bias_exp=np.ones(shape, dtype=np.int32),
                          vth_lo=-3, vth_hi=5)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps_pos.s_out.connect(tlif.a_in)
        sps_neg.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        tlif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        tlif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[2, :] = 1.
        expected_spk_data[(8, 10), :] = -1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_fixed_pm_neg_impulse_du(self):
        """Tests the impulse response of the fixed point ternary LIF neuron
        model with no voltage decay"""
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(-128) * np.ones(shape,
                                                          dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no voltage decay. Current decay is a 12-bit
        # unsigned variable in Loihi hardware. Therefore, du = 2047 is
        # equivalent to (1/2) * (2**12) - 1. The subtracted 1 is added by
        # default in the hardware, via a setting ds_offset, thereby finally
        # giving du = 2048 = 0.5 * 2**12
        # Set up threshold high, such that there are no output spikes. By
        # default the threshold value here is left-shifted by 6.
        tlif = TernaryLIF(shape=shape,
                          du=2047, dv=0,
                          bias_mant=np.zeros(shape, dtype=np.int16),
                          bias_exp=np.ones(shape, dtype=np.int16),
                          vth_lo=(-256) * np.ones(shape, dtype=np.int32),
                          vth_hi=2 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            tlif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(tlif.u.get().astype(np.int32)[0])
        tlif.stop()
        # Gold standard for testing: current decay of 0.5 should halve the
        # current every time-step.
        expected_u_timeseries = [(-1) << (13 - j) for j in range(8)]
        # Gold standard for floating point equivalent of the current,
        # which would be all Loihi-bit-accurate values right shifted by 6 bits
        expected_float_u = [(-1) << (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)
        self.assertListEqual(expected_float_u, np.right_shift(np.array(
            lif_u), 6).tolist())

    def test_fixed_pm_neg_impulse_dv(self):
        """Tests the impulse response of the fixed point ternary LIF neuron
        model with no current decay"""
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(-128) * np.ones(shape,
                                                          dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        # Set up no bias, no current decay. Voltage decay is a 12-bit
        # unsigned variable in Loihi hardware. Therefore, dv = 2048 is
        # equivalent to (1/2) * (2**12).
        # Set up threshold high, such that there are no output spikes.
        # Threshold provided here is left-shifted by 6-bits.
        tlif = TernaryLIF(shape=shape,
                          du=0, dv=2048,
                          bias_mant=np.zeros(shape, dtype=np.int16),
                          bias_exp=np.ones(shape, dtype=np.int16),
                          vth_lo=(-256) * np.ones(shape, dtype=np.int32),
                          vth_hi=2 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(tlif.a_in)
        tlif.s_out.connect(spr.s_in)
        # Configure to run 1 step at a time
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            tlif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(tlif.v.get().astype(np.int32)[0])
        tlif.stop()
        # Gold standard for testing: with a voltage decay of 2048, voltage
        # should integrate from 128<<6 to 255<<6. But it is slightly smaller,
        # because current decay is not exactly 0. Due to the default
        # ds_offset = 1 setting in the hardware, current decay = 1. So
        # voltage is slightly smaller than 128<<6 to 255<<6.
        expected_v_timeseries = [-8192, -12286, -14331, -15351, -15859, -16111,
                                 -16235, -16295]
        # Gold standard for floating point equivalent of the voltage,
        # which would be all Loihi-bit-accurate values right shifted by 6 bits
        expected_float_v = [-128, -192, -224, -240, -248, -252, -254, -255]
        lif_v_float = np.right_shift(np.array(lif_v), 6)
        self.assertListEqual(expected_v_timeseries, lif_v)
        self.assertListEqual(expected_float_v, lif_v_float.tolist())


class TestTLIFReset(unittest.TestCase):
    """Test LIF reset process models"""

    def test_float_model(self):
        """Test float model"""
        num_neurons = 10
        num_steps = 16
        reset_interval = 4
        reset_offset = 3

        lif_reset = LIFReset(shape=(num_neurons,),
                             u=np.arange(num_neurons),
                             du=0,
                             dv=0,
                             vth=100,
                             bias_mant=np.arange(num_neurons) + 1,
                             reset_interval=reset_interval,
                             reset_offset=reset_offset)

        u_logger = io.sink.Read(buffer=num_steps)
        v_logger = io.sink.Read(buffer=num_steps)

        u_logger.connect_var(lif_reset.u)
        v_logger.connect_var(lif_reset.v)

        lif_reset.run(condition=RunSteps(num_steps),
                      run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
        u = u_logger.data.get()
        v = v_logger.data.get()
        lif_reset.stop()

        # Lava timesteps start from t=0. So the first reset offset is missed.
        u_gt_pre = np.vstack([np.arange(num_neurons)] * 2).T
        u_gt_post = np.zeros((num_neurons, num_steps - reset_offset + 1))

        dt = (1 + np.arange(reset_offset - 1)).reshape(1, -1)
        v_gt_pre = np.arange(num_neurons).reshape(-1, 1) * dt \
            + (1 + np.arange(num_neurons)).reshape(-1, 1) * dt
        dt = (1 + np.arange(num_steps - reset_offset + 1) % 4).reshape(1, -1)
        v_gt_post = (1 + np.arange(num_neurons)).reshape(-1, 1) * dt

        self.assertTrue(np.array_equal(u[:, :reset_offset - 1], u_gt_pre))
        self.assertTrue(np.array_equal(u[:, reset_offset - 1:], u_gt_post))
        self.assertTrue(np.array_equal(v[:, :reset_offset - 1], v_gt_pre))
        self.assertTrue(np.array_equal(v[:, reset_offset - 1:], v_gt_post))

    def test_fixed_model(self):
        """Test fixed model"""
        num_neurons = 10
        num_steps = 16
        reset_interval = 4
        reset_offset = 3

        lif_reset = LIFReset(shape=(num_neurons,),
                             u=np.arange(num_neurons),
                             du=-1,
                             dv=0,
                             vth=100,
                             bias_mant=np.arange(num_neurons) + 1,
                             reset_interval=reset_interval,
                             reset_offset=reset_offset)

        u_logger = io.sink.Read(buffer=num_steps)
        v_logger = io.sink.Read(buffer=num_steps)

        u_logger.connect_var(lif_reset.u)
        v_logger.connect_var(lif_reset.v)

        lif_reset.run(condition=RunSteps(num_steps),
                      run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        u = u_logger.data.get()
        v = v_logger.data.get()
        lif_reset.stop()

        # Lava timesteps start from t=0. So the first reset offset is missed.
        u_gt_pre = np.vstack([np.arange(num_neurons)] * 2).T
        u_gt_post = np.zeros((num_neurons, num_steps - reset_offset + 1))

        dt = (1 + np.arange(reset_offset - 1)).reshape(1, -1)
        v_gt_pre = np.arange(num_neurons).reshape(-1, 1) * dt \
            + (1 + np.arange(num_neurons)).reshape(-1, 1) * dt
        dt = (1 + np.arange(num_steps - reset_offset + 1) % 4).reshape(1, -1)
        v_gt_post = (1 + np.arange(num_neurons)).reshape(-1, 1) * dt

        self.assertTrue(np.array_equal(u[:, :reset_offset - 1], u_gt_pre))
        self.assertTrue(np.array_equal(u[:, reset_offset - 1:], u_gt_post))
        self.assertTrue(np.array_equal(v[:, :reset_offset - 1], v_gt_pre))
        self.assertTrue(np.array_equal(v[:, reset_offset - 1:], v_gt_post))


class TestLIFRefractory(unittest.TestCase):
    """Test LIF Refractory process model"""

    def test_float_model(self):
        """Test float model"""
        num_neurons = 2
        num_steps = 8
        refractory_period = 1

        # Two neurons with different biases
        # No Input current provided to make the voltage dependent on the bias
        lif_refractory = LIFRefractory(shape=(num_neurons,),
                                       u=np.zeros(num_neurons),
                                       bias_mant=np.arange(num_neurons) + 1,
                                       bias_exp=np.ones(
                                           (num_neurons,), dtype=float),
                                       vth=4,
                                       refractory_period=refractory_period)

        v_logger = io.sink.Read(buffer=num_steps)
        v_logger.connect_var(lif_refractory.v)

        lif_refractory.run(condition=RunSteps(num_steps),
                           run_cfg=Loihi2SimCfg(select_tag="floating_pt"))

        v = v_logger.data.get()
        lif_refractory.stop()

        # Voltage is expected to remain at reset level for two time steps
        v_expected = np.array([[1, 2, 3, 4, 0, 0, 1, 2],
                               [2, 4, 0, 0, 2, 4, 0, 0]], dtype=float)

        assert_almost_equal(v, v_expected)
