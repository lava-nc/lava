# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List
import numpy as np

from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.magma.core.model.py.model import PyLoihiProcessModel


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag
    """
    def __init__(self, select_tag: str = 'fixed_pt'):
        super(TestRunConfig, self).__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


if __name__ == '__main__':
    num_steps = 10
    shape = (64, 32, 16)
    input = np.random.randint(256, size=shape + (num_steps,))
    input -= 128

    source = SendProcess(data=input)
    sink = ReceiveProcess(shape=(np.prod(shape), ), buffer=num_steps)
    source.out_ports.s_out.flatten().connect(sink.in_ports.a_in)

    run_condition = RunSteps(num_steps=num_steps)
    run_config = TestRunConfig(select_tag='floating_pt')
    sink.run(condition=run_condition, run_cfg=run_config)
    output = sink.data.get()
    sink.stop()

    expected = input.reshape([-1, num_steps])
    print(np.all(output == expected))