# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np
import sys

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
from lava.proc.conv.process import Conv
from lava.proc.conv import utils


verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
TORCH_IS_AVAILABLE = utils.TORCH_IS_AVAILABLE


class ConvRunConfig(RunConfig):
    """Run configuration selects appropriate Conv ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class SendProcess(AbstractProcess):
    """Spike generator process

    Parameters
    ----------
    data: np array
        data to generate spike from. Last dimension is assumed as time.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data = kwargs.pop('data')
        self.data = Var(shape=data.shape, init=data)
        self.s_out = OutPort(shape=data.shape[:-1])  # last dimension is time


class AbstractPySendModel(PyLoihiProcessModel):
    """Template send process model."""
    def run_spk(self):
        buffer = self.data.shape[-1]
        if verbose:
            print(f'{self.current_ts=}')
            data = self.data[..., (self.current_ts - 1) % buffer]
            print(f'Sending data ={data[data!=0]}')
        self.s_out.send(self.data[..., (self.current_ts - 1) % buffer])


@implements(proc=SendProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySendModelFloat(AbstractPySendModel):
    """Float send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=SendProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySendModelFixed(AbstractPySendModel):
    """Fixed point send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


class ReceiveProcess(AbstractProcess):
    """Receive process

    Parameters
    ----------
    shape: tuple
        shape of the process
    buffer: int
        size of data sink buffer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        buffer = kwargs.get('buffer')
        self.shape = shape
        self.a_in = InPort(shape=shape)
        buffer_shape = shape + (buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))


class AbstractPyReceiveModel(PyLoihiProcessModel):
    """Template receive process model."""
    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        data = self.a_in.recv()
        buffer = self.data.shape[-1]
        if verbose:
            print(f'Received data={data[data!=0]}')
        self.data[..., (self.current_ts - 1) % buffer] = data


@implements(proc=ReceiveProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyReceiveModelFloat(AbstractPyReceiveModel):
    """Float receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


@implements(proc=ReceiveProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyReceiveModelFixed(AbstractPyReceiveModel):
    """Fixed point receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)


def setup_conv():
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

    def test_source_sink(self):
        """Test for source-sink process."""
        num_steps = 10
        shape = np.random.randint([128, 128, 16]) + 1
        input = np.random.randint(
            256,
            size=(shape).tolist() + [num_steps]
        ) - 128
        # this should be uncommented after get() flooring issue is merged
        # input = 0.5 * input

        source = SendProcess(data=input)
        sink = ReceiveProcess(shape=tuple(shape), buffer=num_steps)
        source.out_ports.s_out.connect(sink.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = ConvRunConfig(select_tag='floating_pt')
        sink.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        sink.stop()

        self.assertTrue(
            np.all(output == input),
            f'Input and Ouptut do not match.\n'
            f'{output[output!=input]=}\n'
            f'{input[output!=input] =}'
        )

    def test_conv_float(self):
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
            f'{output_gt[output!=output_gt]=}'
        )

    def test_conv_fixed(self):
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
            f'{output_gt[output!=output_gt]=}'
        )
