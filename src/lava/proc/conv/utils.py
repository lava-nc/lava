# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Tuple, Union
import numpy as np
from scipy import signal
from enum import IntEnum, unique

# NOTE: It is known that torch calls inside Lava PyProcess hangs.
# Disabling torch usage inside a Lava CPU process until a fix is found.
# try:
#     import torch
#     import torch.nn.functional as F
#     TORCH_IS_AVAILABLE = True
# except ModuleNotFoundError:
#     TORCH_IS_AVAILABLE = False
TORCH_IS_AVAILABLE = False


@unique
class TensorOrder(IntEnum):
    """Defines how images are represented by tensors.

    Meaning:
        N: number of images in a batch
        H: height of an image
        W: width of an image
        C: number of channels of an image
    """
    NCHW = 1  # default order in PyTorch
    CHWN = 2
    HWCN = 3  # default order in TensorFlow
    NWHC = 4  # default order in Lava


def make_tuple(value: Union[int, Tuple[int, ...]]) -> Tuple[int, int]:
    """Create a tuple of two integers from the given input.

    Parameters
    ----------
    value : int or tuple(int) or tuple(int, int)

    Returns
    -------
    tuple(int, int)
        tuple value of input

    Raises
    ------
    Exception
        if argument value is not 1/2 dimensional.
    """
    if np.isscalar(value):
        return value, value
    elif len(value) == 1:
        return value[0], value[0]
    elif len(value) == 2:
        return value[0], value[1]
    else:
        raise ValueError(
            f"Expected 'value' to be two dimensional."
            f"Got: value = {value}."
        )


def signed_clamp(x: Union[int, float, np.ndarray],
                 bits: int) -> Union[int, float, np.ndarray]:
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


def output_shape(input_shape: Tuple[int, int, int],
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int],
                 dilation: Tuple[int, int]) -> Tuple[int, int, int]:
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


def conv(input_: np.ndarray,
         weight: np.ndarray,
         kernel_size: Tuple[int, int],
         stride: Tuple[int, int],
         padding: Tuple[int, int],
         dilation: Tuple[int, int],
         groups: int) -> np.ndarray:
    """Convolution implementation

    Parameters
    ----------
    input_ : 3 dimensional np array
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
                torch.FloatTensor(input_.transpose([2, 1, 0])),
                dim=0,
            ),
            torch.FloatTensor(
                # torch actually does correlation
                # so flipping the spatial dimension of weight
                # copy() is needed because
                # torch cannot handle negative stride
                weight[:, ::-1, ::-1].transpose([0, 3, 2, 1]).copy()
            ),
            stride=list(stride[::-1]),
            padding=list(padding[::-1]),
            dilation=list(dilation[::-1]),
            groups=groups
        )[0].cpu().data.numpy().transpose([2, 1, 0])
    else:
        output = conv_scipy(
            input_, weight, kernel_size, stride, padding, dilation, groups
        )

    return output.astype(weight.dtype)


def conv_scipy(input_: np.ndarray,
               weight: np.ndarray,
               kernel_size: Tuple[int, int],
               stride: Tuple[int, int],
               padding: Tuple[int, int],
               dilation: Tuple[int, int],
               groups: int) -> np.ndarray:
    """Scipy based implementation of convolution

    Parameters
    ----------
    input_ : 3 dimensional np array
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
    input_shape = input_.shape
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
        input_,
        ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
        mode='constant',
    )

    if input_.shape[-1] % groups != 0:
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
    c_grp = input_.shape[2] // groups
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


def conv_to_sparse(input_shape: Tuple[int, int, int],
                   output_shape: Tuple[int, int, int],
                   kernel: np.ndarray,
                   stride: Tuple[int, int],
                   padding: Tuple[int, int],
                   dilation: Tuple[int, int],
                   group: int,
                   order: TensorOrder = TensorOrder.NWHC) -> Tuple[np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray]:
    """Translate convolution kernel into sparse matrix.

    Parameters
    ----------
    input_shape : tuple of 3 ints
        Shape of input to the convolution.
    output_shape : tuple of 3 ints
        Shape of output from the convolution.
    kernel : numpy array with 4 dimensions
        Convolution kernel. The kernel should have four dimensions. The order
        of kernel tensor is described by ``order`` argument. See Notes for the
        supported orders.
    stride : tuple of 2 ints
        Convolution stride.
    padding : tuple of 2 ints
        Convolution padding.
    dilation : tuple of 2 ints
        Convolution dilation.
    group : int
        Convolution groups.
    order : TensorOrder, optional
        The order of convolution kernel tensor. The default is lava convolution
        order i.e. ``TensorOrder.NWHC``

    Returns
    -------
    np.ndarray
        Destination indices of sparse matrix. It is a linear array of ints.
    np.ndarray
        Source indices of sparse matrix. It is a linear array of ints.
    np.ndarray
        Weight value at non-zero location.

    Raises
    ------
    ValueError
        If tensor order is not supported.
    AssertionError
        if output channels is not divisible by group.
    AssertionError
        if input channels is not divisible by group.

    Notes
    -----

    .. list-table:: Supported tensor dimensions
       :widths: 20, 20, 40
       :header-rows: 1

       * - Input/Output order
         - Kernel order
         - Operation type
       * - WHC
         - NWHC
         - Default Lava order. The operation is convolution.
       * - CHW
         - NCHW
         - Default PyTorch order. The operation in correlation.
       * - HWC
         - HWCN
         - Default Tensorflow order. The operation in correlation.
    """

    if order == TensorOrder.NWHC:  # default order in Lava
        size_xin, size_yin, size_cin = input_shape
        size_xout, size_yout, size_cout = output_shape
        stride_x, stride_y = stride
        pad_x, pad_y = padding
        dilation_x, dilation_y = dilation
        _, conv_x, conv_y, _ = kernel.shape

        def src_id(x: int, y: int, c: int):
            return x * size_yin * size_cin + y * size_cin + c

        def dst_id(x: int, y: int, c: int):
            return x * size_yout * size_cout + y * size_cout + c
    elif order == TensorOrder.NCHW:  # default order in PyTorch
        size_cin, size_yin, size_xin = input_shape
        size_cout, size_yout, size_xout = output_shape
        stride_y, stride_x = stride
        pad_y, pad_x = padding
        dilation_y, dilation_x = dilation
        _, _, conv_y, conv_x = kernel.shape

        def src_id(x: int, y: int, c: int):
            return c * size_yin * size_xin + y * size_xin + x

        def dst_id(x: int, y: int, c: int):
            return c * size_yout * size_xout + y * size_xout + x
        # Convert Tensor order from NCHW to NWHC
        # + flip the spatial dimension because PyTorch actually
        #   does correlation
        kernel = kernel[:, ::-1, ::-1, :].transpose([0, 3, 1, 2])
    elif order == TensorOrder.CHWN:
        size_cin, size_yin, size_xin = input_shape
        size_cout, size_yout, size_xout = output_shape
        stride_y, stride_x = stride
        pad_y, pad_x = padding
        dilation_y, dilation_x = dilation
        _, conv_y, conv_x, _ = kernel.shape

        def src_id(x: int, y: int, c: int):
            return c * size_yin * size_xin + y * size_xin + x

        def dst_id(x: int, y: int, c: int):
            return c * size_yout * size_xout + y * size_xout + x
    elif order == TensorOrder.HWCN:  # default order in TensorFlow
        size_yin, size_xin, size_cin = input_shape
        size_yout, size_xout, size_cout = output_shape
        stride_y, stride_x = stride
        pad_y, pad_x = padding
        dilation_y, dilation_x = dilation
        conv_y, conv_x, *_ = kernel.shape

        def src_id(x: int, y: int, c: int):
            return y * size_xin * size_cin + x * size_cin + c

        def dst_id(x: int, y: int, c: int):
            return y * size_xout * size_cout + x * size_yout + c
        # Convert Tensor order from HWCN to NWHC
        # + flip the spatial dimension because Tensorflow actually
        #   does correlation
        kernel = kernel[::-1, ::-1, :, :].transpose([3, 1, 0, 2])
    else:
        raise ValueError("TensorOrder has been incorrectly specified.")

    # order in WH (XY)
    xx_dst, yy_dst = np.meshgrid(np.arange(size_xout), np.arange(size_yout))
    xx_dst = xx_dst.flatten()
    yy_dst = yy_dst.flatten()
    xx_src = xx_dst * stride_x
    yy_src = yy_dst * stride_y

    srcs = []
    dsts = []
    wgts = []

    for grp in np.arange(group):
        # assuming size_cout and size_cin are absolutely divisible by group
        if size_cin % group != 0:
            raise AssertionError('size_cin must be absolutely divisible by '
                                 f'group. Found {size_cin=} and {group=}')
        if size_cout % group != 0:
            raise AssertionError('size_cout must be absolutely divisible '
                                 f'by group. Found {size_cout=} and {group=}')

        # The strategy here is to connect all the neurons in a spatial location
        # to it's inputs
        c_dst = np.arange(size_cout // group) + grp * (size_cout // group)

        for x_dst, y_dst, x_src, y_src in zip(xx_dst, yy_dst, xx_src, yy_src):
            dst = dst_id(x_dst, y_dst, c_dst)
            dst = dst.astype(int)

            dx, dy = np.meshgrid(np.arange(conv_x), np.arange(conv_y))
            dx = dx.flatten() * dilation_x
            dy = dy.flatten() * dilation_y
            xx = x_src + dx - pad_x
            yy = y_src + dy - pad_y
            valid = np.logical_and(
                np.logical_and(np.greater_equal(xx, 0), np.less(xx, size_xin)),
                np.logical_and(np.greater_equal(yy, 0), np.less(yy, size_yin)),
            )

            for i in range(len(dx)):
                if valid[i]:
                    num_src = size_cin // group
                    c_src = np.arange(num_src) + grp * (num_src)
                    src = src_id(xx[i], yy[i], c_src)
                    weight = kernel[c_dst,
                                    conv_x - int(dx[i] / dilation_x) - 1,
                                    conv_y - int(dy[i] / dilation_y) - 1, :]

                    ss, dd = np.meshgrid(src, dst)

                    srcs.append(ss.flatten())
                    dsts.append(dd.flatten())
                    wgts.append(weight.flatten())

    srcs = np.concatenate(srcs).astype(int)
    dsts = np.concatenate(dsts).astype(int)
    wgts = np.concatenate(wgts)
    return dsts, srcs, wgts
