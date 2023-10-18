# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys
import unittest
import numpy as np
from typing import Tuple

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.bit_check.process import BitCheck
from lava.proc.sdn.process import Sigma, SigmaDelta, ActivationMode
from lava.proc import io

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False


class TestBitCheckModels(unittest.TestCase):
    """Tests for BitCheck Models"""

    def run_test(
        self,
        num_steps: int,
        tag: str = 'fixed_pt',
        bits: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_ = np.sin(0.1 * np.arange(num_steps).reshape(1, -1))
        if tag == 'fixed_pt':
            input_ *= (1 << 12)
            input_ = input_.astype(int)
        input_[:, 1:] -= input_[:, :-1]

        source = io.source.RingBuffer(data=input_)
        sigma = Sigma(shape=(1,))
        sink = io.sink.RingBuffer(shape=sigma.shape, buffer=num_steps)

        source.s_out.connect(sigma.a_in)
        sigma.s_out.connect(sink.a_in)

        debug = 0
        if verbose:
            debug = 1
        bitcheck = BitCheck(shape=sigma.shape, layerid=1, bits=bits, debug=debug)
        bitcheck.ref.connect_var(sigma.sigma)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi1SimCfg(select_tag=tag)

        sigma.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()

        bits_used = bitcheck.bits.get()
        overflowed = bitcheck.overflowed

        sigma.stop()

        return input_, output, bits_used, overflowed

    def test_sigma_decoding_fixed_overflow(self) -> None:
        """Test BitCheck with overflow sigma decode."""
        num_steps = 100
        bitcheck_bits = None
        bitcheck_overflowed = None

        _, _, bitcheck_bits, bitcheck_overflowed = self.run_test(
            num_steps=num_steps,
            tag='fixed_pt',
            bits=12
        )

        if verbose:
            print("bitcheck_overflowed: ", bitcheck_overflowed)
            print("bitcheck_bits: ", bitcheck_bits)
        self.assertTrue(bitcheck_overflowed == 1)
        self.assertTrue(bitcheck_bits == 12)
    
    def test_sigma_decoding_fixed(self) -> None:
        """Test BitCheck no overflow sigma decode."""
        num_steps = 100

        _, _, bitcheck_bits, bitcheck_overflowed  = self.run_test(
            num_steps=num_steps,
            tag='fixed_pt',
            bits=24
        )

        if verbose:
            print("bitcheck_overflowed: ", bitcheck_overflowed)
            print("bitcheck_bits: ", bitcheck_bits)
        self.assertTrue(bitcheck_overflowed == 0)
        self.assertTrue(bitcheck_bits == 24)


class TestBitcheckSigmaDelta(unittest.TestCase):
    """Test BitCheck with sigma delta neurons."""

    def run_test(
        self,
        num_steps: int,
        vth: int,
        act_mode: ActivationMode,
        spike_exp: int,
        state_exp: int,
        cum_error: bool,
        tag: str = 'fixed_pt',
        bits: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_ = np.sin(0.1 * np.arange(num_steps).reshape(1, -1))
        input_ *= (1 << spike_exp + state_exp)
        input_[:, 1:] -= input_[:, :-1]

        source = io.source.RingBuffer(data=input_.astype(int) * (1 << 6))
        sdn = SigmaDelta(
            shape=(1,),
            vth=vth,
            act_mode=act_mode,
            spike_exp=spike_exp,
            state_exp=state_exp,
            cum_error=cum_error
        )
        sink = io.sink.RingBuffer(shape=sdn.shape, buffer=num_steps)

        source.s_out.connect(sdn.a_in)
        sdn.s_out.connect(sink.a_in)

        debug = 0
        if verbose:
            debug = 1
        bitcheck = BitCheck(shape=sdn.shape, layerid=1, bits=bits, debug=debug)
        bitcheck.ref.connect_var(sdn.sigma)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi1SimCfg(select_tag=tag)

        sdn.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        bits_used = bitcheck.bits.get()
        overflowed = bitcheck.overflowed
        sdn.stop()

        input_ = np.cumsum(input_, axis=1)
        output = np.cumsum(output, axis=1)

        return input_, output, bits_used, overflowed

    def test_reconstruction_fixed(self) -> None:
        """Tests BitCheck with fixed point sigma delta reconstruction
        """
        num_steps = 100
        spike_exp = 6
        state_exp = 6
        vth = 10 << (spike_exp + state_exp)
        _, _, bitcheck_bits, bitcheck_overflowed = self.run_test(
            num_steps=num_steps,
            vth=vth,
            act_mode=ActivationMode.UNIT,
            spike_exp=spike_exp,
            state_exp=state_exp,
            cum_error=False,
            bits=24
        )

        if verbose:
            print("bitcheck_overflowed: ", bitcheck_overflowed)
            print("bitcheck_bits: ", bitcheck_bits)
        self.assertTrue(bitcheck_overflowed == 0)
        self.assertTrue(bitcheck_bits == 24)

    def test_reconstruction_fixed_overflow(self) -> None:
        """Tests BitCheck overflow with fixed point sigma delta reconstruction
        """
        num_steps = 100
        spike_exp = 6
        state_exp = 6
        vth = 10 << (spike_exp + state_exp)
        _, _, bitcheck_bits, bitcheck_overflowed  = self.run_test(
            num_steps=num_steps,
            vth=vth,
            act_mode=ActivationMode.UNIT,
            spike_exp=spike_exp,
            state_exp=state_exp,
            cum_error=False,
            bits=12
        )

        if verbose:
            print("bitcheck_overflowed: ", bitcheck_overflowed)
            print("bitcheck_bits: ", bitcheck_bits)
        self.assertTrue(bitcheck_overflowed == 1)
        self.assertTrue(bitcheck_bits == 12)

