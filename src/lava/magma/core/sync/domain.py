# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.sync.protocol import AbstractSyncProtocol


class SyncDomain:
    """
    Specify to run a group of `Processes` using a specific `SyncProtocol`.

    A `SyncProtocol` defines how and when `Processes` are synchronized and
    communication is possible. The `SyncDomain` maps a list of `Processes` to
    a given `SyncProtocol`.

    Parameters
    ----------
    name: str
        Name of the SyncDomain.
    protocol: AbstractSyncProtocol
        SyncProtocol the `Processes` are mapped to.
    processes: ty.List[AbstractProcess]
        List of `Processes` to run in the given SyncProtocol.

    """

    def __init__(
        self,
        name: str,
        protocol: AbstractSyncProtocol = None,
        processes: ty.List["AbstractProcess"] = None,  # noqa: F821
    ):
        self.name = name
        self.protocol = protocol
        self.processes = processes
        if processes is None:
            self.processes = []

    def set_protocol(self, protocol: AbstractSyncProtocol):
        if not isinstance(protocol, AbstractSyncProtocol):
            raise AssertionError
        self.protocol = protocol

    def add_process(
        self, process: ty.Union["AbstractProcess",  # noqa: F821
                                ty.List["AbstractProcess"]]  # noqa: F821
    ):
        process = [process]
        self.processes += process
