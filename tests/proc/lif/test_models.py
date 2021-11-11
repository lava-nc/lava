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
from lava.proc.lif.process import LIF


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


class VecRecvProcess(AbstractProcess):
    """
    Sink process that receives arbitrary vectors

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
class PyVecSendModel1(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, float)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.current_ts - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('fixed_pt')
class PyVecSendModel2(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int16, precision=16)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=16)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.current_ts - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('floating_pt')
class PySpkRecvModel1(PyLoihiProcessModel):
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
class PySpkRecvModel2(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.current_ts - 1, :] = spk_in


class TestLIFProcessModels(unittest.TestCase):
    """Tests for all ProcessModels of LIF"""
    def test_float_pm_no_decay(self):
        """
        Set up bias = 1 * 2**1 = 2. and threshold = 4. Without any current
        and voltage decay and in the absence of external input, bias driven
        neurons spike every 2nd time-step.
        """
        shape = (10,)
        num_steps = 10
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=np.float),
                             send_at_times=np.ones((num_steps,), dtype=np.bool))
        lif = LIF(shape=shape,
                  du=0.,
                  dv=0.,
                  bias=np.ones(shape, dtype=np.float),
                  bias_exp=np.ones(shape, dtype=np.float),
                  vth=4.)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif.run(condition=rcnd, run_cfg=rcfg)
        spk_data_through_run = spr.spk_data.get()
        lif.stop()
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[1:10:2, :] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_impulse_du(self):
        """
        Send a single spike of magnitude 128 with a current decay of 0.5 and
        voltage decay of 0. Current state variable decays by half every
        time-step.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=np.float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        lif = LIF(shape=shape,
                  du=0.5,  # decay_u = 1/2
                  dv=0,
                  bias=np.zeros(shape, dtype=np.float),
                  bias_exp=np.ones(shape, dtype=np.float),
                  vth=256.)  # high thr, no spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(lif.u.get()[0])
        lif.stop()
        expected_u_timeseries = [2. ** (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)

    def test_float_pm_impulse_dv(self):
        """
        Send a single spike of magnitude 128 with a voltage decay of 0.5 and
        current decay of 0. Voltage state variable increases from 128. to 255.
        with steps of 64., 32., 16., etc.
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=(2 ** 7) * np.ones(shape,
                                                            dtype=np.float),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        lif = LIF(shape=shape,
                  du=0,
                  dv=0.5,  # decay_v = 1/2
                  bias=np.zeros(shape, dtype=np.float),
                  bias_exp=np.ones(shape, dtype=np.float),
                  vth=256.)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(lif.v.get()[0])
        lif.stop()
        expected_v_timeseries = [128., 192., 224., 240., 248., 252., 254., 255.]
        print(expected_v_timeseries)
        self.assertListEqual(expected_v_timeseries, lif_v)

    def test_bitacc_pm_no_decay(self):
        """
        Set up bias = 2 * 2**6 = 128 and threshold = 8<<6 Without any current
        and voltage decay and in the absence of external input, bias driven
        neurons spike every 4th time-step.
        """
        shape = (10,)
        num_steps = 10
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=np.int16),
                             send_at_times=np.ones((num_steps,), dtype=np.bool))
        lif = LIF(shape=shape,
                  du=0, dv=0,
                  bias=2 * np.ones(shape, dtype=np.int32),
                  bias_exp=6 * np.ones(shape, dtype=np.int32),
                  vth=8)
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif.run(condition=rcnd, run_cfg=rcfg)
        spk_data_through_run = spr.spk_data.get()
        lif.stop()
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[3:10:4, :] = 1
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_bitacc_pm_impulse_du(self):
        """
        Send a single activation of magnitude 128 with a current decay of
        2047 and voltage decay of 0. Current state variable decays by half every
        time-step. Actual current values, compared to the floating point test
        above are left-shifted by 6 bits due to bit-accurate fixed point
        implementation with respect to Loihi hardware
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        lif = LIF(shape=shape,
                  du=2047,  # decay_u = 1/2
                  dv=0,
                  bias=np.zeros(shape, dtype=np.int16),
                  bias_exp=np.ones(shape, dtype=np.int16),
                  vth=256 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_u = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_u.append(lif.u.get().astype(np.int32)[0])
        lif.stop()
        expected_u_timeseries = [1 << (13 - j) for j in range(8)]
        expected_float_u = [1 << (7 - j) for j in range(8)]
        self.assertListEqual(expected_u_timeseries, lif_u)
        self.assertListEqual(expected_float_u, np.right_shift(np.array(
            lif_u), 6).tolist())

    def test_bitacc_pm_impulse_dv(self):
        """
        Send a single activation of magnitude 128 with a voltage decay of
        2048 and current decay of 0. Voltage state variable increases from
        128<<6 to 254<<6. Actual voltage values, compared to the floating point
        test above are left-shifted by 6 bits due to bit-accurate fixed point
        implementation with respect to Loihi hardware. Further the
        corrected voltage values (right-shifted back by 6) are smaller by 1
        due to default hardware settings in Loihi, which set minimum value of
        decay_u = 1 (and not 0).
        """
        shape = (1,)  # a single neuron
        num_steps = 8
        # send activation of 128. at timestep = 1
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=128 * np.ones(shape, dtype=np.int32),
                             send_at_times=np.array([True, False, False,
                                                     False, False, False,
                                                     False, False]))
        lif = LIF(shape=shape,
                  du=0,
                  dv=2048,  # decay_v = 1/2
                  bias=np.zeros(shape, dtype=np.int16),
                  bias_exp=np.ones(shape, dtype=np.int16),
                  vth=256 * np.ones(shape, dtype=np.int32))
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.out_ports.s_out.connect(lif.in_ports.a_in)
        lif.out_ports.s_out.connect(spr.in_ports.s_in)
        rcnd = RunSteps(num_steps=1)
        rcfg = LifRunConfig(select_tag='fixed_pt')
        lif_v = []
        # Run 1 timestep at a time and collect state variable u
        for j in range(num_steps):
            lif.run(condition=rcnd, run_cfg=rcfg)
            lif_v.append(lif.v.get().astype(np.int32)[0])
        lif.stop()
        expected_v_timeseries = [8192, 12286, 14331, 15351, 15859, 16111,
                                 16235, 16295]
        expected_float_v = [128, 192, 224, 240, 248, 252, 254, 255]
        lif_v_float = np.right_shift(np.array(lif_v), 6)
        lif_v_float[1:] += 1
        self.assertListEqual(expected_v_timeseries, lif_v)
        self.assertListEqual(expected_float_v, lif_v_float.tolist())
