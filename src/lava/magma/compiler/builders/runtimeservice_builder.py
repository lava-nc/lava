# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import logging
import typing as ty

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspRecvPort, CspSendPort
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.enums import LoihiVersion
from lava.magma.runtime.runtime_services.runtime_service import \
    AbstractRuntimeService

try:
    from lava.magma.runtime.runtime_services.nxsdk_runtime_service import \
        NxSdkRuntimeService
except ImportError:
    class NxSdkRuntimeService:
        pass


class RuntimeServiceBuilder:
    """RuntimeService builders instantiate and initialize a RuntimeService.

    Parameters
    ----------
    rs_class: AbstractRuntimeService class of the runtime service to build.
    sync_protocol: AbstractSyncProtocol Synchronizer class that
                   implements a protocol in a domain.
    """

    def __init__(
            self,
            rs_class: ty.Type[AbstractRuntimeService],
            protocol: ty.Type[AbstractSyncProtocol],
            runtime_service_id: int,
            model_ids: ty.List[int],
            loihi_version: ty.Type[LoihiVersion],
            loglevel: int = logging.WARNING,
            compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None,
            *args,
            **kwargs
    ):
        self.rs_class = rs_class
        self.sync_protocol = protocol
        self.rs_args = args
        self.rs_kwargs = kwargs
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        self._compile_config = compile_config
        self._runtime_service_id = runtime_service_id
        self._model_ids: ty.List[int] = model_ids
        self.csp_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.csp_proc_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_proc_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.loihi_version: ty.Type[LoihiVersion] = loihi_version

    @property
    def runtime_service_id(self):
        """Return runtime service id."""
        return self._runtime_service_id

    def set_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set CSP Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_recv_port.update({port.name: port})

    def set_csp_proc_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set CSP Process Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_proc_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_proc_recv_port.update({port.name: port})

    def build(self) -> AbstractRuntimeService:
        """Build the runtime service

        Returns
        -------
        A concreate instance of AbstractRuntimeService
        [PyRuntimeService or NxSdkRuntimeService]
        """

        self.log.debug("RuntimeService Class: " + str(self.rs_class))
        nxsdk_rts = False
        if self.rs_class == NxSdkRuntimeService:
            rs = self.rs_class(
                self.sync_protocol,
                loihi_version=self.loihi_version,
                loglevel=self.log.level,
                compile_config=self._compile_config,
                **self.rs_kwargs
            )
            nxsdk_rts = True
            self.log.debug("Initilized NxSdkRuntimeService")
        else:
            rs = self.rs_class(protocol=self.sync_protocol)
            self.log.debug("Initilized PyRuntimeService")
        rs.runtime_service_id = self._runtime_service_id
        rs.model_ids = self._model_ids

        if not nxsdk_rts:
            for port in self.csp_proc_send_port.values():
                if "service_to_process" in port.name:
                    rs.service_to_process.append(port)

            for port in self.csp_proc_recv_port.values():
                if "process_to_service" in port.name:
                    rs.process_to_service.append(port)

            self.log.debug("Setup 'RuntimeService <--> Rrocess; ports")

        for port in self.csp_send_port.values():
            if "service_to_runtime" in port.name:
                rs.service_to_runtime = port

        for port in self.csp_recv_port.values():
            if "runtime_to_service" in port.name:
                rs.runtime_to_service = port

        self.log.debug("Setup 'Runtime <--> RuntimeService' ports")

        return rs
