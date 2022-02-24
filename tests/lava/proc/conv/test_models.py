# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from typing import Dict, List, Tuple, Type, Union
import unittest
import numpy as np
import sys

from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.conv.process import Conv
from lava.proc.conv import utils
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc.io.sink import RingBuffer as ReceiveProcess


verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
TORCH_IS_AVAILABLE = utils.TORCH_IS_AVAILABLE
np.random.seed(9933)


class ConvRunConfig(RunConfig):
    """Run configuration selects appropriate Conv ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, select_tag: str = 'fixed_pt'):
        super().__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


def setup_conv() -> Tuple[
    Type[Conv],
    Tuple[int, int, int],
    Tuple[int, int, int],
    Dict[str, Union[np.ndarray, Tuple[int, int], int]]
]:
    """Sets up random convolution setting."""
    # conv parameter setup
    groups = np.random.randint(4) + 1
    in_channels = (np.random.randint(8) + 1) * groups
    out_channels = (np.random.randint(8) + 1) * groups
    kernel_size = np.random.randint([9, 9]) + 1
    stride = np.random.randint([5, 5]) + 1
    padding = np.random.randint([5, 5])
    dilation = np.random.randint([4, 4]) + 1
    weight_dims = [
        out_channels,
        kernel_size[0], kernel_size[1],
        in_channels // groups
    ]
    weight = np.random.randint(256, size=weight_dims) - 128

    # input needs to be a certain size
    # to make sure the output dimension is never negative
    input_shape = tuple(
        (np.random.randint([128, 128]) + kernel_size * dilation).tolist()
        + [in_channels]
    )
    output_shape = utils.output_shape(
        input_shape, out_channels, kernel_size, stride, padding, dilation
    )

    conv = Conv(
        input_shape=input_shape,
        weight=weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    params = {
        'weight': weight,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups,
    }

    return conv, input_shape, output_shape, params


class TestConvProcessModels(unittest.TestCase):
    """Tests for all ProcessModels of Conv"""

    @unittest.skip
    def test_conv_float(self) -> None:
        """Test for float conv process."""
        num_steps = 10
        utils.TORCH_IS_AVAILABLE = False

        conv, input_shape, output_shape, params = setup_conv()
        input = np.random.random(input_shape + (num_steps,))
        input = (input > 0.8)

        source = SendProcess(data=input)
        sink = ReceiveProcess(shape=output_shape, buffer=num_steps)

        source.out_ports.s_out.connect(conv.in_ports.s_in)
        conv.out_ports.a_out.connect(sink.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = ConvRunConfig(select_tag='floating_pt')
        conv.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        conv.stop()

        utils.TORCH_IS_AVAILABLE = TORCH_IS_AVAILABLE

        output_gt = np.zeros_like(output)
        for t in range(output.shape[-1]):
            output_gt[..., t] = utils.conv(input[..., t], **params)

        error = np.abs(output - output_gt).mean()

        if error >= 1e-6:
            print(f'{input.shape=}')
            print(f'{output.shape=}')
            print(f'{params["weight"].shape=}')
            print(f'{params["kernel_size"]=}')
            print(f'{params["stride"]=}')
            print(f'{params["padding"]=}')
            print(f'{params["dilation"]=}')
            print(f'{params["groups"]=}')

        self.assertTrue(
            error < 1e-6,
            f'Output and ground truth do not match.\n'
            f'{output[output!=output_gt]   =}\n'
            f'{output_gt[output!=output_gt]=}\n'
        )

    @unittest.skip
    def test_conv_fixed(self) -> None:
        """Test for fixed point conv process."""
        num_steps = 10
        utils.TORCH_IS_AVAILABLE = False

        conv, input_shape, output_shape, params = setup_conv()
        input = np.random.random(input_shape + (num_steps,))
        input = (input > 0.8)

        source = SendProcess(data=input)
        sink = ReceiveProcess(shape=output_shape, buffer=num_steps)

        source.out_ports.s_out.connect(conv.in_ports.s_in)
        conv.out_ports.a_out.connect(sink.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = ConvRunConfig(select_tag='fixed_pt')
        conv.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        conv.stop()

        utils.TORCH_IS_AVAILABLE = TORCH_IS_AVAILABLE

        output_gt = np.zeros_like(output)
        for t in range(output.shape[-1]):
            output_gt[..., t] = utils.conv(input[..., t], **params)
        output_gt = utils.signed_clamp(output_gt, bits=24)

        error = np.abs(output - output_gt).mean()

        if error >= 1e-6:
            print(f'{input.shape=}')
            print(f'{output.shape=}')
            print(f'{params["weight"].shape=}')
            print(f'{params["kernel_size"]=}')
            print(f'{params["stride"]=}')
            print(f'{params["padding"]=}')
            print(f'{params["dilation"]=}')
            print(f'{params["groups"]=}')

        self.assertTrue(
            error < 1e-6,
            f'Output and ground truth do not match.\n'
            f'{output[output!=output_gt]   =}\n'
            f'{output_gt[output!=output_gt]=}\n'
        )
