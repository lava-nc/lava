# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty

from lava.magma.core.model.interfaces import AbstractPortMessage, \
    PortMessageFormat


class PortMessage(AbstractPortMessage):
    pass


def enum_to_np(value: ty.Union[int, float],
               d_type: type = np.int32) -> np.array:
    """
    Helper function to convert an int (or EnumInt) or a float to a single value
    np array so as to pass it via the message passing framework. The dtype of
    the np array is specified by d_type with the default of np.int32.

    :param value: value to be converted to a 1-D array
    :param d_type: type of the converted np array
    :return: np array with the value
    """

    return np.array([value], dtype=d_type)


def enum_equal(a: np.array, b: np.array) -> bool:
    """
    Helper function to compare two np arrays created by enum_to_np.

    :param a: 1-D array created by enum_to_np
    :param b: 1-D array created by enum_to_np
    :return: True if the two arrays are equal
    """
    return a[0] == b[0]


def enum_to_message(value: ty.Union[int, float],
                    d_type: type = np.int32,
                    msg_fmt: PortMessageFormat
                    = PortMessageFormat.MGMT) -> PortMessage:
    """Helper function to convert an int (or EnumInt) to a
    AbstractPortMessage to pass it via the message passing framework
    Parameters
    ----------
    value : int
        value to be converted to AbstractPortMessage
    Returns
    -------
    AbstractPortMessage
    """
    data = np.array([value], dtype=d_type)
    return PortMessage(
        msg_fmt,
        data.size,
        data
    )


class MGMT_COMMAND:
    """
    Signifies the Mgmt Command being sent between two actors. These may be
    between runtime and runtime_service or the runtime_service
    and process model.
    """

    RUN = enum_to_message(0)
    """Signifies a RUN command for 0 timesteps from one actor to another. Any
    non negative integer signifies a run command"""
    STOP = enum_to_message(-1)
    """Signifies a STOP command from one actor to another"""
    PAUSE = enum_to_message(-2)
    """Signifies a PAUSE command from one actor to another"""


class REQ_TYPE:
    """
    Signifies type of request
    """
    GET = enum_to_message(0)
    """Read a variable"""
    SET = enum_to_message(1)
    """Write to a variable"""


class MGMT_RESPONSE:
    """Signifies the response to a Mgmt command. This response can be sent
    by any actor upon receiving a Mgmt command"""

    DONE = enum_to_message(0)
    """Signfies Ack or Finished with the Command"""
    TERMINATED = enum_to_message(-1)
    """Signifies Termination"""
    ERROR = enum_to_message(-2)
    """Signifies Error raised"""
    PAUSED = enum_to_message(-3)
    """Signifies Execution State to be Paused"""
