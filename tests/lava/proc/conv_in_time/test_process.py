# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import os
import numpy as np
from lava.proc.conv_in_time.process import ConvInTime
from lava.proc import io

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.conv import utils

if utils.TORCH_IS_AVAILABLE:
    import torch
    import torch.nn as nn
    compare = True
    # In this case, the test compares against random torch ground truth
else:
    compare = False
    # In this case, the test compares against saved torch ground truth


class TestConvInTimeProcess(unittest.TestCase):
    """Tests for Conv class"""
    def test_init(self) -> None:
        """Tests instantiation of Conv In Time"""
        num_steps = 10
        kernel_size = 3
        n_in = 2
        n_out = 5
        if compare:
            spike_input = np.random.choice(
                [0, 1],
                size=(n_in, num_steps))
            weights = np.random.randint(256, size=[kernel_size,
                                                   n_out,
                                                   n_in]) - 128
        else:
            spike_input = np.load(os.path.join(os.path.dirname(__file__),
                                               "gts/spike_input.npy"))
            weights = np.load(os.path.join(os.path.dirname(__file__),
                                           "gts/q_weights.npy"))
        sender = io.source.RingBuffer(data=spike_input)
        conv_in_time = ConvInTime(weights=weights, name='conv_in_time')

        receiver = io.sink.RingBuffer(
            shape=(n_out,),
            buffer=num_steps + 1)

        sender.s_out.connect(conv_in_time.s_in)
        conv_in_time.a_out.connect(receiver.a_in)

        run_condition = RunSteps(num_steps=num_steps + 1)
        run_cfg = Loihi1SimCfg(select_tag="floating_pt")

        conv_in_time.run(condition=run_condition, run_cfg=run_cfg)
        output = receiver.data.get()
        conv_in_time.stop()

        if compare:
            tensor_input = torch.tensor(spike_input, dtype=torch.float32)
            tensor_weights = torch.tensor(weights, dtype=torch.float32)
            conv_layer = nn.Conv1d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=kernel_size, bias=False)
            # Permute the weights to match the torch format
            conv_layer.weight = nn.Parameter(tensor_weights.permute(1, 2, 0))
            torch_output = conv_layer(
                tensor_input.unsqueeze(0)).squeeze(0).detach().numpy()
        else:
            torch_output = np.load(os.path.join(os.path.dirname(__file__),
                                                "gts/torch_output.npy"))

        self.assertEqual(output.shape, (n_out, num_steps + 1))
        # After kernel_size timesteps,
        # the output should be the same as the torch output
        assert np.allclose(output[:, kernel_size:], torch_output)
