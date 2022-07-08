# Copyright (C) 2022 Intel Corporation
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
from lava.proc.delay.process import Delay

from lava.tests.lava.proc.dense.test_models import (
    VecSendandRecvProcess,
    VecRecvProcess
)


class DelayRunConfig(RunConfig):
    """
    Run configuration selects appropriate Delay ProcessModel based on tag:
        floating point precision
    Currently, only floating point precision is available.
    """

    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError("No legal ProcessModel found.")


class TestDelayProcessModelFloat(unittest.TestCase):
    """Tests for floating point ProcessModels of Delay"""

    def test_float_pm_buffer(self):
        """
        Tests floating point Delay ProcessModel connectivity and temporal
        dynamics. All input 'neurons' from the VecSendandRcv fire
        once at time t=4, and only 1 connection weight in the Dense
        Process is non-zero. The non-zero connection should have an
        activation of 1 at timestep t=5.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(
            shape=(shape[1],),
            num_steps=num_steps,
            vec_to_send=vec_to_send,
            send_at_times=send_at_times
        )
        # Set up Delay Process with a single non-zero connection weight at
        # entry [2, 2] of the connectivity mat.
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        delay = Delay(shape=shape, weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(delay.s_in)
        delay.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DelayRunConfig(select_tag='floating_pt')
        delay.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        delay.stop()
        # Gold standard for the test
        # a_out will be equal to 1 at timestep 4, because the dendritic
        #  accumulators work on inputs from the previous timestep.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 2] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_delay(self):
        """
        Tests floating point Delay ProcessModel connectivity and temporal
        dynamics when delays are included. All input 'neurons' from the
        VecSendandRcv fire once at time t=4, and only 1 connection weight
        in the Dense Process is non-zero. The non-zero connection also has
        a non-zero delay. The non-zero connection should have an activation
        of 1 at timestep t=8 (three-timestep delay).
        """
        shape = (3, 4)
        num_steps = 9
        # Set up external input to emulate every neuron spiking once on
        # timestep 4
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(
            shape=(shape[1],),
            num_steps=num_steps,
            vec_to_send=vec_to_send,
            send_at_times=send_at_times
        )
        # Set up Delay Process with a single non-zero connection weight
        # and a delay of 3 at entry [2, 2] of the connectivity mat.
        weights = np.zeros(shape, dtype=float)
        weights[2, 2] = 1
        delays = np.zeros(shape, dtype=float)
        delays[2, 2] = 3
        delay = Delay(shape=shape, weights=weights, delays=delays)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(delay.s_in)
        delay.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DelayRunConfig(select_tag='floating_pt')
        delay.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        delay.stop()
        # Gold standard for the test
        # a_out will be equal to 1 at timestep 7, because the dendritic
        #  accumulators work on inputs from the previous timestep.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[7, 2] = 1.
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
        sps = VecSendandRecvProcess(
            shape=(shape[1],),
            num_steps=num_steps,
            vec_to_send=vec_to_send,
            send_at_times=send_at_times
        )
        # Set up a Delay Process where all input layer neurons project to a
        # single output layer neuron.
        weights = np.zeros(shape, dtype=float)
        weights[2, :] = [2, -3, 4, -5]
        delay = Delay(shape=shape, weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(delay.s_in)
        delay.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DelayRunConfig(select_tag='floating_pt')
        delay.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        delay.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neuron 3 will be equal to -2 = 2 - 3 + 4 - 5 at timestep 5
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, 2] = -2
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_fan_out(self):
        """
        Tests floating point Delay ProcessModel dendritic accumulation
        behavior when the fan-out of a projecting neuron is greater than 1.
        """
        shape = (3, 4)
        num_steps = 6
        # Set up external input to emulate every neuron spiking once on
        # timestep t=4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(False, (num_steps,))
        send_at_times[3] = True
        sps = VecSendandRecvProcess(
            shape=(shape[1],),
            num_steps=num_steps,
            vec_to_send=vec_to_send,
            send_at_times=send_at_times
        )
        # Set up a Delay Process where a single input layer neuron projects to
        # all output layer neurons.
        weights = np.zeros(shape, dtype=float)
        weights[:, 2] = [3, 4, 5]
        delay = Delay(shape=shape, weights=weights)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(delay.s_in)
        delay.a_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DelayRunConfig(select_tag='floating_pt')
        delay.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        delay.stop()
        # Gold standard for the test
        # Expected behavior is that a_out corresponding to output
        # neurons 1-3 will be equal to 3, 4, and 5, respectively,
        # at timestep 5.
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4, :] = [3, 4, 5]
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))

    def test_float_pm_recurrence(self):
        """
        Tests that floating Delay ProcessModel has non-blocking dynamics for
        recurrent connectivity architectures.
        """
        shape = (3, 3)
        num_steps = 6
        # Set up external inputs to emulate every neuron spiking once on
        # timestep 4.
        vec_to_send = np.ones((shape[1],), dtype=float)
        send_at_times = np.repeat(True, (num_steps,))
        sps = VecSendandRecvProcess(
            shape=(shape[1],),
            num_steps=num_steps,
            vec_to_send=vec_to_send,
            send_at_times=send_at_times
        )
        # Set up Delay Process with fully connected recurrent connectivity
        # architecture
        weights = np.ones(shape, dtype=float)
        delay = Delay(shape=shape, weights=weights)
        # Receive neuron spikes
        sps.s_out.connect(delay.s_in)
        delay.a_out.connect(sps.a_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = DelayRunConfig(select_tag='floating_pt')
        delay.run(condition=rcnd, run_cfg=rcfg)
        delay.stop()
