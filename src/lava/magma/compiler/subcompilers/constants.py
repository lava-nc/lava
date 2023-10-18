# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from enum import IntEnum

COUNTERS_PER_EMBEDDED_CORE = 900
EMBEDDED_CORE_COUNTER_START_INDEX = 33

NUM_VIRTUAL_CORES_L2 = 128
NUM_VIRTUAL_CORES_L3 = 120

MAX_EMBEDDED_CORES_PER_CHIP = 3

SPIKE_BLOCK_CORE = 0xFFFF


class EMBEDDED_ALLOCATION_ORDER(IntEnum):
    NORMAL = 1
    """Allocate embedded cores in normal order 0, 1, 2"""
    REVERSED = -1
    """Allocate embedded cores in reverse order 2, 1, 0. This is useful in
    situations in case of certain tasks which take longer than others and
    need to be scheduled on embedded core 0 to ensure nxcore does not stop
    communicating on channels"""
