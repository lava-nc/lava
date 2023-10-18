# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from enum import IntEnum, auto
from dataclasses import dataclass
import typing as ty
import numpy as np
import warnings

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort


class SendFull(IntEnum):
    """Enum specifying a channel's sender policy when the buffer is full.

    SendFull.BLOCKING : The sender blocks when the buffer is full.
    SendFull.NON_BLOCKING_DROP : The sender does not block when the buffer is
    full. Instead, it issues a warning and does not send the data.
    """
    BLOCKING = auto()
    NON_BLOCKING_DROP = auto()


def send_full_blocking(src_port: CspSendPort, data: np.ndarray) -> None:
    """Send data through the src_port, when using SendFull.BLOCKING.

    If the channel buffer is full, this method blocks.

    Parameters
    ----------
    src_port : CspSendPort
        Source port through which the data is sent.
    data : np.ndarray
        Data to be sent.
    """
    src_port.send(data)


def send_full_non_blocking_drop(src_port: CspSendPort, data: np.ndarray) -> \
        None:
    """Send data through the src_port, when using SendFull.NON_BLOCKING_DROP.

    If the channel buffer is full, the data is not sent, a warning is issued,
    and this method does not block.

    Parameters
    ----------
    src_port : CspSendPort
        Source port through which the data is sent.
    data : np.ndarray
        Data to be sent.
    """
    if src_port.probe():
        src_port.send(data)
    else:
        warnings.warn("Send buffer is full. Dropping items ...")


SEND_FULL_FUNCTIONS = {
    SendFull.BLOCKING: send_full_blocking,
    SendFull.NON_BLOCKING_DROP: send_full_non_blocking_drop
}


class ReceiveEmpty(IntEnum):
    """Enum specifying a channel's receiver policy when the buffer is empty.

    ReceiveEmpty.BLOCKING : The receiver blocks when the buffer is empty.
    ReceiveEmpty.NON_BLOCKING_ZEROS : The receiver does not block when the
    buffer is empty. Instead, it receives zeros.
    """
    BLOCKING = auto()
    NON_BLOCKING_ZEROS = auto()


def receive_empty_blocking(dst_port: CspRecvPort, _: np.ndarray) -> np.ndarray:
    """Receive data through the dst_port, when using ReceiveEmpty.BLOCKING.

    This method is blocking.

    Parameters
    ----------
    dst_port : CspRecvPort
        Destination port through which the data is received.
    _ : np.ndarray
        Zeros array.

    Returns
    ----------
    data : np.ndarray
        Data received.
    """
    return dst_port.recv()


def receive_empty_non_blocking_zeros(_: CspRecvPort, zeros: np.ndarray) \
        -> np.ndarray:
    """Receive data through the dst_port, when using
    ReceiveEmpty.NON_BLOCKING_ZEROS.

    This method returns zeros.

    Parameters
    ----------
    _ : CspRecvPort
        Destination port.
    zeros : np.ndarray
        Zeros array to be returned.

    Returns
    ----------
    data : np.ndarray
        Data received (zeros).
    """
    return zeros


RECEIVE_EMPTY_FUNCTIONS = {
    ReceiveEmpty.BLOCKING: receive_empty_blocking,
    ReceiveEmpty.NON_BLOCKING_ZEROS: receive_empty_non_blocking_zeros
}


class ReceiveNotEmpty(IntEnum):
    """Enum specifying a channel's receiver policy when the buffer is not empty.

    ReceiveNotEmpty.FIFO : The receiver receives a single item, being the
    oldest one in the buffer.
    ReceiveNotEmpty.ACCUMULATE : The receiver receives all items,
    accumulating them.
    """
    FIFO = auto()
    ACCUMULATE = auto()


def receive_not_empty_fifo(dst_port: CspRecvPort, _: np.ndarray, __: int) \
        -> np.ndarray:
    """Receive data through the dst_port, when using ReceiveNotEmpty.FIFO.

    This method receives a single item from the dst_port and returns it.

    Parameters
    ----------
    dst_port : CspRecvPort
        Destination port through which the data is received.
    _ : np.ndarray
        Zeros array.
    __ : int
        Number of elements in the buffer.

    Returns
    ----------
    data : np.ndarray
        Data received.
    """
    return dst_port.recv()


def receive_not_empty_accumulate(dst_port: CspRecvPort, zeros: np.ndarray,
                                 elements_in_buffer: int) -> np.ndarray:
    """Receive data through the dst_port, when using ReceiveNotEmpty.ACCUMULATE.

    This method receives a all items from the dst_port, accumulates them and
    returns the result.

    Parameters
    ----------
    dst_port : CspRecvPort
        Port through which the data is received.
    zeros : np.ndarray
        Zeros array to initialize the accumulator.
    elements_in_buffer : int
        Number of elements in the buffer.

    Returns
    ----------
    data : np.ndarray
        Data received (accumulated).
    """
    data = zeros

    for _ in range(elements_in_buffer):
        data += dst_port.recv()

    return data


RECEIVE_NOT_EMPTY_FUNCTIONS = {
    ReceiveNotEmpty.FIFO: receive_not_empty_fifo,
    ReceiveNotEmpty.ACCUMULATE: receive_not_empty_accumulate
}


@dataclass
class ChannelConfig:
    """Dataclass wrapping the different channel configuration parameters."""
    send_full: SendFull = SendFull.BLOCKING
    receive_empty: ReceiveEmpty = ReceiveEmpty.BLOCKING
    receive_not_empty: ReceiveNotEmpty = ReceiveNotEmpty.FIFO

    def get_send_full_function(self) -> ty.Callable:
        return SEND_FULL_FUNCTIONS[self.send_full]

    def get_receive_empty_function(self) -> ty.Callable:
        return RECEIVE_EMPTY_FUNCTIONS[self.receive_empty]

    def get_receive_not_empty_function(self) -> ty.Callable:
        return RECEIVE_NOT_EMPTY_FUNCTIONS[self.receive_not_empty]


def validate_shape(shape: ty.Tuple[int, ...]):
    """Validate the shape parameter.

    Parameters
    ----------
    shape : tuple
        Shape to validate.
    """
    if not isinstance(shape, tuple):
        raise TypeError("Expected <shape> to be of type tuple. Got "
                        f"<shape> = {shape}.")

    for s in shape:
        if not np.issubdtype(type(s), int):
            raise TypeError("Expected all elements of <shape> to be of "
                            f"type int. Got <shape> = {shape}.")
        if s <= 0:
            raise ValueError("Expected all elements of <shape> to be "
                             f"strictly positive. Got <shape> = {shape}.")


def validate_buffer_size(buffer_size: int):
    """Validate the buffer_size parameter.

    Parameters
    ----------
    buffer_size : int
        Buffer size to validate.
    """
    if not np.issubdtype(type(buffer_size), int):
        raise TypeError("Expected <buffer_size> to be of type int. Got "
                        f"<buffer_size> = {buffer_size}.")
    if buffer_size <= 0:
        raise ValueError("Expected <buffer_size> to be strictly positive. Got "
                         f"<buffer_size> = {buffer_size}.")


def validate_channel_config(channel_config: ChannelConfig) -> None:
    """Validate the channel_config parameter.

    Parameters
    ----------
    channel_config : ChannelConfig
        Channel configuration to validate.
    """
    if not isinstance(channel_config, ChannelConfig):
        raise TypeError(
            "Expected <channel_config> to be of type "
            "ChannelConfig. Got "
            f"<channel_config> = {channel_config}.")

    if not isinstance(channel_config.send_full, SendFull):
        raise TypeError(
            "Expected <channel_config>.send_full "
            "to be of type SendFull. Got "
            "<channel_config>.send_full = "
            f"{channel_config.send_full}.")

    if not isinstance(channel_config.receive_empty, ReceiveEmpty):
        raise TypeError(
            "Expected <channel_config>.receive_empty "
            "to be of type ReceiveEmpty. Got "
            "<channel_config>.receive_empty = "
            f"{channel_config.receive_empty}.")

    if not isinstance(channel_config.receive_not_empty, ReceiveNotEmpty):
        raise TypeError(
            "Expected <channel_config>.receive_not_empty "
            "to be of type ReceiveNotEmpty. Got "
            "<channel_config>.receive_not_empty = "
            f"{channel_config.receive_not_empty}.")
