# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

"""Defines message tokens for Actions (Commands) and Responses. Also defines
helper functions to convert scalar values to these message tokens."""

import typing as ty
import numpy as np


def enum_to_np(value: ty.Union[int, float],
               d_type: type = np.float64) -> np.array:
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


class MGMT_COMMAND:
    """
    Signifies the Mgmt Command being sent between two actors. These may be
    between runtime and runtime_service or the runtime_service
    and process model.
    """

    RUN = enum_to_np(0)
    """Signifies a RUN command for 0 timesteps from one actor to another. Any
    non negative integer signifies a run command"""
    STOP = enum_to_np(-1)
    """Signifies a STOP command from one actor to another"""
    PAUSE = enum_to_np(-2)
    """Signifies a PAUSE command from one actor to another"""
    GET_DATA = enum_to_np(-3)
    """Signifies Read a variable"""
    SET_DATA = enum_to_np(-4)
    """Signifies Write a variable"""


class MGMT_RESPONSE:
    """Signifies the response to a Mgmt command. This response can be sent
    by any actor upon receiving a Mgmt command"""

    DONE = enum_to_np(0)
    """Signfies Ack or Finished with the Command"""
    TERMINATED = enum_to_np(-1)
    """Signifies Termination"""
    ERROR = enum_to_np(-2)
    """Signifies Error raised"""
    PAUSED = enum_to_np(-3)
    """Signifies Execution State to be Paused"""
    REQ_PAUSE = enum_to_np(-4)
    """Signifies Request of PAUSE"""
    REQ_STOP = enum_to_np(-5)
    """Signifies Request of STOP"""
    SET_COMPLETE = enum_to_np(-6)
    """Signifies Completion of Set Var"""
