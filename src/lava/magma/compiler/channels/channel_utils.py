# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
import numpy as np
from lava.magma.compiler.channels.pypychannel import __PyPyChannel


def __create_pypy_mgmt_channel(smm, name) -> __PyPyChannel:
    """
    Helper function to create a python to python Mgmt channel. This is typically
    backed by shared memory. Shared memory needs to be managed by the creator.
    :param smm: Shared Memory Manager
    :param name: Name of the Mgmt Channel. Only Mgmt Commands are sent on this.
    :return: __PyPyChannel Channel Handler
    """
    channel = __PyPyChannel(
        smm=smm, name=name, shape=(1,), dtype=np.int32, size=8
    )
    return channel
