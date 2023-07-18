# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import unittest
import numpy as np
import scipy
from lava.proc.conv import utils

if utils.TORCH_IS_AVAILABLE:
    import torch
    import torch.nn.functional as F
    compare = True
    # in this case, the test compares against torch ground truth
else:
    compare = False
    # in this case, the test only checks for error during
    # utils.conv calculation


class TestConv(unittest.TestCase):
    def test_conv(self) -> None:
        """Test convolution implementation"""
        prng = np.random.RandomState(8534)
        for _ in range(10):  # testing with 10 random combinations
            groups = prng.randint(4) + 1
            in_channels = (prng.randint(8) + 1) * groups
            out_channels = (prng.randint(8) + 1) * groups
            kernel_size = prng.randint([9, 9]) + 1
            stride = prng.randint([5, 5]) + 1
            padding = prng.randint([5, 5])
            dilation = prng.randint([4, 4]) + 1
            weight_dims = [out_channels,
                           kernel_size[0], kernel_size[1],
                           in_channels // groups]
            weights = prng.randint(256, size=weight_dims) - 128
            input_ = prng.random(
                (
                    # input needs to be a certain size
                    # to make sure the output dimension is never negative
                    prng.randint([128, 128]) + kernel_size * dilation
                ).tolist()
                + [in_channels]
            )

            out = utils.conv_scipy(input_, weights, kernel_size,
                                   stride, padding, dilation, groups)

            if compare:  # if torch is available, compare against it.
                out_gt = F.conv2d(
                    torch.unsqueeze(  # torch expects a batch dimension NCHW
                        torch.FloatTensor(input_.transpose([2, 1, 0])),
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
                    print(f'{input_.shape=}')
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
                )

    def test_conv_saved_data(self) -> None:
        """Test convolution implementation against saved data."""
        prng = np.random.RandomState(8534)
        for i in range(10):  # testing with 10 random combinations
            gt_data = np.load(os.path.dirname(os.path.abspath(__file__))
                              + f'/ground_truth/gt_conv_paris_{i}.npz')
            out = utils.conv_scipy(gt_data['input'],
                                   gt_data['weights'],
                                   gt_data['kernel_size'],
                                   gt_data['stride'],
                                   gt_data['padding'],
                                   gt_data['dilation'],
                                   gt_data['groups'])
            out_gt = gt_data['out_gt']
            error = np.abs(out - out_gt).mean()
            self.assertTrue(
                error < 1e-3,
                f'Conv calculation does not match with torch ground truth.'
                f'Found\n'
                f'{out.flatten()[:50] = }\n'
                f'{out_gt.flatten()[:50] = }\n'
            )

    def test_conv_to_sparse(self) -> None:
        """Tests translation of a conv to a sparse layer"""
        prng = np.random.RandomState(8534)
        for _ in range(10):
            groups = prng.randint(4) + 1
            in_channels = (prng.randint(4) + 1) * groups
            out_channels = (prng.randint(8) + 1) * groups
            kernel_size = prng.randint([5, 5]) + 1
            stride = prng.randint([2, 2]) + 1
            padding = prng.randint([2, 2])
            dilation = prng.randint([2, 2]) + 1
            weight_dims = [out_channels,
                           kernel_size[0], kernel_size[1],
                           in_channels // groups]
            weights = prng.randint(256, size=weight_dims) - 128
            input = prng.random(
                (
                    # input needs to be a certain size
                    # to make sure the output dimension is never negative
                    prng.randint([16, 16]) + kernel_size * dilation
                ).tolist()
                + [in_channels]
            )

            output_gt = utils.conv_scipy(input, weights, kernel_size,
                                         stride, padding, dilation, groups)

            dst, src, wgt = utils.conv_to_sparse(input_shape=input.shape,
                                                 output_shape=output_gt.shape,
                                                 kernel=weights,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation,
                                                 group=groups)
            sparse_shape = (np.prod(output_gt.shape), np.prod(input.shape))
            sparse_wgt = scipy.sparse.csc_matrix((wgt, (dst, src)),
                                                 shape=sparse_shape)

            output = sparse_wgt.dot(input.reshape(-1, 1))
            error = np.abs(output.flatten() - output_gt.flatten()).sum()

            if error >= 1e-6:
                print(f'{input.shape=}')
                print(f'{output_gt.shape=}')
                print(f'{weights.shape=}')
                print(f'{stride=}')
                print(f'{padding=}')
                print(f'{dilation=}')
                print(f'{groups=}')
                print(f'{groups=}')
            self.assertTrue(error < 1e-6,
                            f'Conv as sparse calculation does not match the '
                            f'direct convolution output. Found\n'
                            f'{output.flatten()[:10]=}\n'
                            f'{output_gt.flatten()[:10]=}\n')
