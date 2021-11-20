# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import os
import unittest
import numpy as np
from lava.proc.conv import utils

try:
    import torch
    import torch.nn.functional as F
    compare = True
    # in this case, the test compares against torch ground truth
except ModuleNotFoundError:
    compare = False
    # in this case, the test only checks for error during
    # utils.conv calculation

# TODO: add tensorflow support

seed = np.random.randint(1000)
np.random.seed(seed)


class TestConv(unittest.TestCase):
    def test_conv(self) -> None:
        """Test convolution implementation"""
        for _ in range(10):  # testing with 10 random combinations
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
            weights = np.random.randint(256, size=weight_dims) - 128
            input = np.random.random(
                (
                    # input needs to be a certain size
                    # to make sure the output dimension is never negative
                    np.random.randint([128, 128]) + kernel_size * dilation
                ).tolist()
                + [in_channels]
            )

            out = utils.conv_scipy(
                input, weights, kernel_size, stride, padding, dilation, groups
            )

            if compare:  # if torch is available, compare against it.
                out_gt = F.conv2d(
                    torch.unsqueeze(  # torch expects a batch dimension NCHW
                        torch.FloatTensor(input.transpose([2, 1, 0])),
                        dim=0,
                    ),
                    torch.FloatTensor(
                        # torch acutally does correlation
                        # so flipping the spatial dimension of weight
                        # copy() is needed because
                        # torch cannot handle negative stride
                        weights[:, ::-1, ::-1].transpose([0, 3, 2, 1]).copy()
                    ),
                    stride=stride[::-1].tolist(),
                    padding=padding[::-1].tolist(),
                    dilation=dilation[::-1].tolist(),
                    groups=groups
                )[0].cpu().data.numpy().transpose([2, 1, 0])

                error = np.abs(out - out_gt).mean()
                if error >= 1e-3:  # small eps to account for float/double calc
                    # Setting failed! Print out the dimensions for debugging.
                    print(f'{input.shape=}')
                    print(f'{weights.shape=}')
                    print(f'{kernel_size=}')
                    print(f'{stride=}')
                    print(f'{padding=}')
                    print(f'{dilation=}')
                    print(f'{groups=}')
                    print(f'{out.shape=}')
                    print(f'{out_gt.shape=}')
                self.assertTrue(
                    error < 1e-3,
                    f'Conv calculation does not match with torch ground truth.'
                    f'Found\n'
                    f'{out.flatten()[:50] = }\n'
                    f'{out_gt.flatten()[:50] = }\n'
                    f'{seed = }'
                )

    def test_conv_saved_data(self) -> None:
        """Test convolution implementation against saved data."""
        for i in range(10):  # testing with 10 random combinations
            gt_data = np.load(
                os.path.dirname(os.path.abspath(__file__))
                + f'/ground_truth/gt_conv_paris_{i}.npz'
            )
            out = utils.conv_scipy(
                gt_data['input'],
                gt_data['weights'],
                gt_data['kernel_size'],
                gt_data['stride'],
                gt_data['padding'],
                gt_data['dilation'],
                gt_data['groups']
            )
            out_gt = gt_data['out_gt']
            error = np.abs(out - out_gt).mean()
            self.assertTrue(
                error < 1e-3,
                f'Conv calculation does not match with torch ground truth.'
                f'Found\n'
                f'{out.flatten()[:50] = }\n'
                f'{out_gt.flatten()[:50] = }\n'
            )
