from enum import IntEnum, auto
from dataclasses import dataclass
import numpy as np
import warnings
import typing as ty

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort


class ChannelSendBufferFull(IntEnum):
    BLOCKING = auto()
    NON_BLOCKING_DROP = auto()


class ChannelRecvBufferEmpty(IntEnum):
    BLOCKING = auto()
    NON_BLOCKING_ZEROS = auto()


class ChannelRecvBufferNotEmpty(IntEnum):
    FIFO = auto()
    ACCUMULATE = auto()


@dataclass
class ChannelConfig:
    send_buffer_full: ChannelSendBufferFull = \
        ChannelSendBufferFull.BLOCKING
    recv_buffer_empty: ChannelRecvBufferEmpty = \
        ChannelRecvBufferEmpty.BLOCKING
    recv_buffer_not_empty: ChannelRecvBufferNotEmpty = \
        ChannelRecvBufferNotEmpty.FIFO


def validate_shape(shape):
    if not isinstance(shape, tuple):
        raise TypeError("Expected <shape> to be of type tuple. Got "
                        f"<shape> = {shape}.")

    for s in shape:
        if not isinstance(s, int):
            raise TypeError("Expected all elements of <shape> to be of "
                            f"type int. Got <shape> = {shape}.")
        if s <= 0:
            raise ValueError("Expected all elements of <shape> to be "
                             f"strictly positive. Got <shape> = {shape}.")


def validate_dtype(dtype: ty.Union[ty.Type, np.dtype]) -> None:
    if not isinstance(dtype, (type, np.dtype)):
        raise TypeError("Expected <dtype> to be of type type or np.dtype. "
                        f"Got <dtype> = {dtype}.")

def validate_size(size):
    if not isinstance(size, int):
        raise TypeError("Expected <size> to be of type int. Got <size> = "
                        f"{size}.")
    if size <= 0:
        raise ValueError("Expected <size> to be strictly positive. Got "
                         f"<size> = {size}.")


def validate_channel_config(
        channel_config: ChannelConfig) -> None:
    if not isinstance(channel_config, ChannelConfig):
        raise TypeError(
            "Expected <channel_config> to be of type "
            "ChannelConfig. Got "
            f"<channel_config> = {channel_config}.")

    if not isinstance(channel_config.send_buffer_full,
                      ChannelSendBufferFull):
        raise TypeError(
            "Expected <channel_config>.send_buffer_full "
            "to be of type ChannelSendBufferFull. Got "
            "<channel_config>.send_buffer_full = "
            f"{channel_config.send_buffer_full}.")

    if not isinstance(channel_config.recv_buffer_empty,
                      ChannelRecvBufferEmpty):
        raise TypeError(
            "Expected <channel_config>.recv_buffer_empty "
            "to be of type ChannelRecvBufferEmpty. Got "
            "<channel_config>.recv_buffer_empty = "
            f"{channel_config.recv_buffer_empty}.")

    if not isinstance(channel_config.recv_buffer_empty,
                      ChannelRecvBufferEmpty):
        raise TypeError(
            "Expected <channel_config>.recv_buffer_not_empty "
            "to be of type ChannelRecvBufferEmpty. Got "
            "<channel_config>.recv_buffer_not_empty = "
            f"{channel_config.recv_buffer_not_empty}.")


def validate_send_data(data: np.ndarray, out_port_shape: tuple) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("Expected <data> to be of type np.ndarray. Got "
                        f"<data> = {data}")

    if data.shape != out_port_shape:
        raise ValueError("Expected <data>.shape to be equal to shape of "
                         f"OutPort. Got <data>.shape = {data.shape} and "
                         f"<out_port>.shape = {out_port_shape}.")


def send_data_blocking(src_port: CspSendPort, data: np.ndarray) -> None:
    src_port.send(data)


def send_data_non_blocking_drop(src_port: CspSendPort, data: np.ndarray) -> \
        None:
    if src_port.probe():
        src_port.send(data)
    else:
        warnings.warn("Send buffer is full. Dropping items ...")


def recv_empty_blocking(dst_port: CspRecvPort, **kwargs) -> np.ndarray:
    return dst_port.recv()

def recv_empty_non_blocking_zeros(zeros: np.ndarray, **kwargs) \
        -> np.ndarray:
    return zeros

def recv_not_empty_fifo(dst_port: CspRecvPort, **kwargs) -> np.ndarray:
    return dst_port.recv()

def recv_not_empty_accumulate(dst_port: CspRecvPort, zeros: np.ndarray,
                              elements_in_queue: int) -> \
        np.ndarray:
    data = zeros

    for _ in range(elements_in_queue):
        data += dst_port.recv()

    return data
