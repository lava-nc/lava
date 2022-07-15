# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from enum import IntEnum


class SpikeType(IntEnum):
    """Types of Spike Encoding"""
    BINARY_SPIKE = 1
    LONG_SPIKE = 2
    GRADED8_SPIKE = 3
    GRADED16_SPIKE = 4
    GRADED24_SPIKE = 5
    POP16_SPIKE = 6
    POP32_SPIKE = 7
