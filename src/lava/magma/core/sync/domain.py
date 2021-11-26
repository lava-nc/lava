# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.sync.protocol import AbstractSyncProtocol


class SyncDomain:
    """"""

    def __init__(
        self,
        name: str,
        protocol: AbstractSyncProtocol = None,
        processes: ty.List["AbstrctProcess"] = None,  # noqa: F821
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
        # TODO: isinstance with string does not work
        if True:  # TODO: isinstance(process, 'AbstractProcess'):
            process = [process]
        if isinstance(process, list):
            for item in process:
                pass
                # assert isinstance(item, AbstractProcess)
        self.processes += process
