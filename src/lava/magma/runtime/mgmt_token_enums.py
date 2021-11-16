# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.magma.core.model.interfaces import AbstractPortMessage, PortMessageFormat


def enum_to_np(value: int) -> np.array:
    """
    Helper function to convert an int (or EnumInt) to a single value np array
    so as to pass it via the message passing framework

    :param value: value to be converted to a 1-D array
    :return: np array with the value
    """
    return np.array([value], dtype=np.int32)


def enum_to_message(value: int) -> AbstractPortMessage:
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
    data = np.array([value], dtype=np.int32)
    return AbstractPortMessage(
                PortMessageFormat.MGMT,
                data.size,
                data
    )


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


class REQ_TYPE:
    """
    Signifies type of request
    """
    GET = enum_to_np(0)
    """Read a variable"""
    SET = enum_to_np(1)
    """Write to a variable"""


class MGMT_RESPONSE:
    """Signifies the response to a Mgmt command. This response can be sent
    by any actor upon receiving a Mgmt command"""

    DONE = enum_to_np(0)
    """Signfies Ack or Finished with the Command"""
    TERMINATED = enum_to_np(-1)
    """Signifies Termination"""
    PAUSED = enum_to_np(-2)
    """Signifies Execution State to be Paused"""
