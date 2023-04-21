# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from enum import IntEnum
from lava.magma.runtime.mgmt_token_enums import enum_to_np


class LoihiVersion(IntEnum):
    """Enumerator of different Loihi Versions."""
    N2 = 2
    N3 = 3


class LoihiPhase:
    """Enumerator of Lava Loihi phases"""
    SPK = enum_to_np(1)
    PRE_MGMT = enum_to_np(2)
    LRN = enum_to_np(3)
    POST_MGMT = enum_to_np(4)
    HOST = enum_to_np(5)


class NxSdkPhase:
    """Enumerator phases in which snip can run in nxcore."""

    EMBEDDED_INIT = 1
    """INIT Phase of Embedded Snip. This executes only once."""
    EMBEDDED_SPIKING = 2
    """SPIKING Phase of Embedded Snip."""
    EMBEDDED_PRELEARN_MGMT = 3
    """Pre-Learn Management Phase of Embedded Snip."""
    EMBEDDED_MGMT = 4
    """Management Phase of Embedded Snip."""
    HOST_PRE_EXECUTION = 5
    """Host Pre Execution Phase for Host Snip."""
    HOST_POST_EXECUTION = 6
    """Host Post Execution Phase for Host Snip."""
    HOST_CONCURRENT_EXECUTION = 7
    """Concurrent Execution for Host Snip."""
    EMBEDDED_USER_CMD = 8
    """Any User Command to execute during embedded execution.
    (Internal Use Only)"""
    EMBEDDED_REMOTE_MGMT = 9
    """A management phase snip triggered remotely"""
