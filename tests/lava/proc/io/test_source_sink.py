# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np

from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc.io.sink import RingBuffer as ReceiveProcess


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag
    """
    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, _, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class TestSendReceive(unittest.TestCase):
    """Tests for all SendProces and ReceiveProcess."""

    def test_source_sink(self):
        """Test whatever is being sent form source is received at sink."""
        num_steps = 10
        shape = np.random.randint([128, 128, 16]) + 1
        input = np.random.randint(256, size=(shape).tolist() + [num_steps])
        input -= 128
        # input = 0.5 * input

        source = SendProcess(data=input)
        sink = ReceiveProcess(shape=tuple(shape), buffer=num_steps)
        source.out_ports.s_out.connect(sink.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='floating_pt')
        sink.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        sink.stop()

        self.assertTrue(
            np.all(output == input),
            f'Input and Ouptut do not match.\n'
            f'{output[output!=input]=}\n'
            f'{input[output!=input] =}'
        )
