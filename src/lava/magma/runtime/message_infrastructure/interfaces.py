# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from enum import IntEnum


class ChannelType(IntEnum):
    """Type of a channel given the two process models"""

    PyPy = 0
    CPy = 1
    PyC = 2
    CNc = 3
    NcC = 4
    CC = 3
    NcNc = 5
    NcPy = 6
    PyNc = 7
