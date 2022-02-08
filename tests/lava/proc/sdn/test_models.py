# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import sys
import unittest
import numpy as np
from typing import Dict, List, Tuple, Type, Union

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.sdn.process import SigmaDelta, ACTIVATION_MODE
from lava.proc import io


verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False


class TestSigmaDeltaModels(unittest.TestCase):
    """Tests for sigma delta neuron"""

    def run_test(
        self,
        num_steps: int,
        vth: int,
        act_mode: ACTIVATION_MODE,
        wgt_exp: int,
        state_exp: int,
        cum_error: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input = np.sin(0.1 * np.arange(num_steps).reshape(1, -1))
        input *= (1 << wgt_exp + state_exp)
        input[:, 1:] -= input[:, :-1]

        source = io.source.RingBuffer(data=input.astype(int) * (1 << 6))
        sdn = SigmaDelta(
            shape=(1,),
            vth=vth,
            act_mode=act_mode,
            wgt_exp=wgt_exp,
            state_exp=state_exp,
            cum_error=cum_error
        )
        sink = io.sink.RingBuffer(shape=sdn.shape, buffer=num_steps)

        source.s_out.connect(sdn.a_in)
        sdn.s_out.connect(sink.a_in)
        
        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')

        sdn.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        sdn.stop()

        input = np.cumsum(input, axis=1)
        output = np.cumsum(output, axis=1)

        return input, output

    def test_reconstruction(self) -> None:
        num_steps = 100
        vth = 10
        wgt_exp = 6
        state_exp = 6
        input, output = self.run_test(
            num_steps=num_steps,
            vth=vth,
            act_mode=ACTIVATION_MODE.Unit,
            wgt_exp=wgt_exp,
            state_exp=state_exp,
            cum_error=False,
        )

        error = np.abs(input - output).max()

        if verbose:
            print(f'Max abs error = {error}')
        self.assertTrue(error < vth * (1 << wgt_exp))

    def test_reconstruction_cum_error(self) -> None:
        num_steps = 100
        vth = 10
        wgt_exp = 6
        state_exp = 6
        input, output = self.run_test(
            num_steps=num_steps,
            vth=vth,
            act_mode=ACTIVATION_MODE.Unit,
            wgt_exp=wgt_exp,
            state_exp=state_exp,
            cum_error=True,
        )

        error = np.abs(input - output).max()

        if verbose:
            print(f'Max abs error = {error}')
        self.assertTrue(error < vth * (1 << wgt_exp))

    def test_reconstruction_relu(self) -> None:
        num_steps = 100
        vth = 10
        wgt_exp = 6
        state_exp = 6
        input, output = self.run_test(
            num_steps=num_steps,
            vth=vth,
            act_mode=ACTIVATION_MODE.ReLU,
            wgt_exp=wgt_exp,
            state_exp=state_exp,
            cum_error=False,
        )

        error = np.abs(np.maximum(input, 0) - output).max()

        if verbose:
            print(f'Max abs error = {error}')
        self.assertTrue(error < vth * (1 << wgt_exp))


if __name__ == '__main__':
    num_steps = 100
    vth = 10
    wgt_exp = 6
    state_exp = 6
    act_mode = ACTIVATION_MODE.ReLU
    cum_error = False,
    input = np.sin(0.1 * np.arange(num_steps).reshape(1, -1))
    input *= (1 << wgt_exp + state_exp)
    input[:, 1:] -= input[:, :-1]

    source = io.source.RingBuffer(data=input.astype(int) * (1 << 6))
    sdn = SigmaDelta(
        shape=(1,),
        vth=vth,
        act_mode=act_mode,
        wgt_exp=wgt_exp,
        state_exp=state_exp,
        cum_error=cum_error
    )
    sink = io.sink.RingBuffer(shape=sdn.shape, buffer=num_steps)

    # source.s_out.connect(sdn.a_in)
    sdn.s_out.connect(sink.a_in)
    
    run_condition = RunSteps(num_steps=num_steps)
    run_config = Loihi1SimCfg(select_tag='fixed_pt')

    sdn.run(condition=run_condition, run_cfg=run_config)
    output = sink.data.get()
    sdn.stop()

    input = np.cumsum(input, axis=1)
    output = np.cumsum(output, axis=1)

