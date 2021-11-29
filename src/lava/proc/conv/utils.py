# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Tuple, Union
import numpy as np
from scipy import signal

try:
    import torch
    import torch.nn.functional as F
    TORCH_IS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_IS_AVAILABLE = False

# TODO: add tensorflow support


def get_tuple(
    kwargs: dict,
    name: str, default: Union[int, Tuple[int, ...]]
) -> Tuple[int, int]:
    """Get tuple value from keyword argument.

    Parameters
    ----------
    kwargs : dict
        keyword argument
    name : str
        name/key string
    default : int
        default value

    Returns
    -------
    tuple of two int
        tuple value of input

    Raises
    ------
    Exception
        if argument value is not 1/2 dimensional.
    """
    shape = kwargs.get(name, default)
    if np.isscalar(shape):
        return (shape, shape)
    elif len(shape) == 1:
        return (shape[0], shape[0])
    elif len(shape) == 2:
        return (shape[0], shape[1])
    else:
        raise ValueError(
            f'Expected {name} to be two dimensional.'
            f'Found {name} = {shape}.'
        )


def signed_clamp(
    x: Union[int, float, np.ndarray],
    bits: int
) -> Union[int, float, np.ndarray]:
    """clamps as if input is a signed value within the precision of bits.

    Parameters
    ----------
    x : int, float, np array
        input
    bits : int
        number of bits for the variable

    Returns
    -------
    same type as x
        clamped value
    """
    base = 1 << bits
    return (x + base // 2) % base - base // 2  # signed value clamping


def output_shape(
    input_shape: Tuple[int, int, int],
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int]
) -> Tuple[int, int, int]:
    """Calculates the output shape of convolution operation.

    Parameters
    ----------
    input_shape : 3 element tuple, list, or array
        shape of input to convolution in XYZ/WHC format.
    out_channels : int
        number of output channels.
    kernel_size : 2 element tuple, list, or array
        convolution kernel size in XY/WH format.
    stride : 2 element tuple, list, or array
        convolution stride in XY/WH format.
    padding : 2 element tuple, list, or array
        convolution padding in XY/WH format.
    dilation : 2 element tuple, list, or array
        dilation of convolution kernel in XY/WH format.

    Returns
    -------
    tuple of 3 ints
        shape of convolution output in XYZ/WHC format.

    Raises
    ------
    Exception
        for invalid x convolution dimension.
    Exception
        for invalid y convolution dimension.
    """
    x_out = np.floor(
        (
            input_shape[0] + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1) - 1
        ) / stride[0] + 1
    ).astype(int)
    y_out = np.floor(
        (
            input_shape[1] + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1) - 1
        ) / stride[1] + 1
    ).astype(int)

    if x_out < 1:
        print(f'{input_shape=}')
        print(f'{out_channels=}')
        print(f'{kernel_size=}')
        print(f'{stride=}')
        print(f'{padding=}')
        print(f'{dilation=}')
        print(x_out, y_out, out_channels)
        raise Exception(
            f'Found output x dimension (={x_out}) to be less than 1.'
            f'Check your convolution sizes.'
        )
    if y_out < 1:
        print(f'{input_shape=}')
        print(f'{out_channels=}')
        print(f'{kernel_size=}')
        print(f'{stride=}')
        print(f'{padding=}')
        print(f'{dilation=}')
        print(x_out, y_out, out_channels)
        raise Exception(
            f'Found output y dimension (={y_out}) to be less than 1.'
            f'Check your convolution sizes.'
        )

    return x_out, y_out, out_channels


def conv(
    input: np.ndarray,
    weight: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int
) -> np.ndarray:
    """Convolution implementation

    Parameters
    ----------
    input : 3 dimensional np array
        convolution input.
    weight : 4 dimensional np array
        convolution kernel weight.
    kernel_size : 2 element tuple, list, or array
        convolution kernel size in XY/WH format.
    stride : 2 element tuple, list, or array
        convolution stride in XY/WH format.
    padding : 2 element tuple, list, or array
        convolution padding in XY/WH format.
    dilation : 2 element tuple, list, or array
        dilation of convolution kernel in XY/WH format.
    groups : int
        number of convolution groups.

    Returns
    -------
    3 dimensional np array
        convolution output
    """
    if TORCH_IS_AVAILABLE:
        # with torch.no_grad():  # this seems to cause problems
        output = F.conv2d(
            torch.unsqueeze(  # torch expects a batch dimension NCHW
                torch.FloatTensor(input.transpose([2, 1, 0])),
                dim=0,
            ),
            torch.FloatTensor(
                # torch actually does correlation
                # so flipping the spatial dimension of weight
                # copy() is needed because
                # torch cannot handle negative stride
                weight[:, ::-1, ::-1].transpose([0, 3, 2, 1]).copy()
            ),
            stride=stride[::-1].tolist(),
            padding=padding[::-1].tolist(),
            dilation=dilation[::-1].tolist(),
            groups=groups
        )[0].cpu().data.numpy().transpose([2, 1, 0])
    else:
        output = conv_scipy(
            input, weight, kernel_size, stride, padding, dilation, groups
        )

    return output.astype(weight.dtype)


def conv_scipy(
    input: np.ndarray,
    weight: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int
) -> np.ndarray:
    """Scipy based implementation of convolution

    Parameters
    ----------
    input : 3 dimensional np array
        convolution input.
    weight : 4 dimensional np array
        convolution kernel weight.
    kernel_size : 2 element tuple, list, or array
        convolution kernel size in XY/WH format.
    stride : 2 element tuple, list, or array
        convolution stride in XY/WH format.
    padding : 2 element tuple, list, or array
        convolution padding in XY/WH format.
    dilation : 2 element tuple, list, or array
        dilation of convolution kernel in XY/WH format.
    groups : int
        number of convolution groups.

    Returns
    -------
    3 dimensional np array
        convolution output
    """
    input_shape = input.shape
    output = np.zeros(
        output_shape(
            input_shape, weight.shape[0],
            kernel_size, stride, padding, dilation
        )
    )

    dilated_weight = np.zeros([
        weight.shape[0],
        dilation[0] * (kernel_size[0] - 1) + 1,
        dilation[1] * (kernel_size[1] - 1) + 1,
        weight.shape[-1]
    ])
    dilated_weight[:, ::dilation[0], ::dilation[1], :] = weight

    input_padded = np.pad(
        input,
        ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
        mode='constant',
    )

    if input.shape[-1] % groups != 0:
        raise Exception(
            f'Expected number of in_channels to be divisible by group.'
            f'Found {weight.shape[3] = } and {groups = }.'
        )
    if output.shape[-1] % groups != 0:
        raise Exception(
            f'Expected number of out_channels to be divisible by group.'
            f'Found {weight.shape[0] = } and {groups = }.'
        )

    k_grp = output.shape[2] // groups
    c_grp = input.shape[2] // groups
    for g in range(groups):
        for k in range(k_grp):
            for c in range(c_grp):
                temp = signal.convolve2d(
                    input_padded[:, :, c + g * c_grp],
                    dilated_weight[k + g * k_grp, :, :, c],
                    mode='valid'
                )
                output[:, :, k + g * k_grp] += temp[::stride[0], ::stride[1]]
    return output
