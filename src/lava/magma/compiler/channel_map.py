# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import itertools
import typing as ty
from collections import defaultdict
from dataclasses import dataclass

from lava.magma.compiler.compiler_graphs import ProcGroup
from lava.magma.compiler.utils import PortInitializer
from lava.magma.core.process.ports.ports import AbstractPort
from lava.magma.core.process.ports.ports import AbstractSrcPort, AbstractDstPort
from lava.magma.core.process.process import AbstractProcess


@dataclass(eq=True, frozen=True)
class PortPair:
    src: AbstractSrcPort
    dst: AbstractDstPort


@dataclass
class Payload:
    multiplicity: int
    tiling: ty.Optional[ty.Tuple[int, ...]] = None
    src_port_initializer: PortInitializer = None
    dst_port_initializer: PortInitializer = None


def lmt_init_id():
    return -1


class ChannelMap(dict):
    """The ChannelMap is used by the SubCompilers during compilation to
    communicate how they are planning to partition Processes onto their
    available resources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initializers_lookup = dict()
        self._lmt_allocation_dict: ty.Dict[int, int] = defaultdict(lmt_init_id)

    def __setitem__(
        self, key: PortPair, value: Payload, dict_setitem=dict.__setitem__
    ):
        self._validate_item(key, value)
        dict_setitem(self, key, value)

    @staticmethod
    def _validate_item(key: PortPair, value: Payload) -> None:
        if not isinstance(key, PortPair):
            raise TypeError(
                f"Key of ChannelMap should be of type PortPair, "
                f"got type {type(key)}."
            )
        if not isinstance(value, Payload):
            raise TypeError(
                f"Value of ChannelMap should be of type Payload, "
                f"got type {type(value)}."
            )

    @property
    def lmt_allocation_dict(self) -> ty.Dict[int, int]:
        return self._lmt_allocation_dict

    @classmethod
    def from_proc_groups(self,
                         proc_groups: ty.List[ProcGroup]) -> "ChannelMap":
        """Initializes a ChannelMap from a list of process ProcGroups
        extracting the ports from every process group.

        For every port pair in the process groups a PortPair will be created and
        set as a key on the ChannelMap object which inherits from standard dict,
        the values are initialized with default Payload having multiplicity
        of 1.

        Every subcompiler will then update the multiplicity according to its
        partitioning process. Payload also include fields for
        parameterization of convolutional processes.

        Parameters
        ----------
        proc_groups: a list of ProcessGroups each multiple processes.

        Returns
        -------
        channel_map: A ChannelMap object initialized with the PortPairs found in
        the list of process groups given as input.
        """
        port_pairs = self._get_port_pairs_from_proc_groups(proc_groups)
        channel_map = ChannelMap()
        for port_pair in port_pairs:
            channel_map[port_pair] = Payload(multiplicity=1)
        return channel_map

    @classmethod
    def _get_port_pairs_from_proc_groups(cls,
                                         proc_groups: ty.List[ProcGroup]):
        """Loop over processes connectivity and get all connected port pairs."""
        processes = list(itertools.chain.from_iterable(proc_groups))
        port_pairs = []
        for src_process in processes:
            src_ports = (
                src_process.out_ports.members + src_process.ref_ports.members
            )
            for src_port in src_ports:
                dst_ports = src_port.get_dst_ports()
                for dst_port in dst_ports:
                    if cls._is_leaf_process_port(dst_port, processes):
                        port_pairs.append(PortPair(src=src_port, dst=dst_port))
        return port_pairs

    @staticmethod
    def _is_leaf_process_port(dst_port, processes):
        dst_process = dst_port.process
        return True if dst_process in processes else False

    def set_port_initializer(self,
                             port: AbstractPort,
                             port_initializer: PortInitializer):
        if port in self._initializers_lookup.keys():
            raise AssertionError(
                "An initializer for this port has already " "been assigned."
            )
        self._initializers_lookup[port] = port_initializer

    def get_port_initializer(self, port):
        return self._initializers_lookup[port]

    def has_port_initializer(self, port) -> bool:
        return port in self._initializers_lookup

    def write_to_cache(self,
                       cache_object: ty.Dict[ty.Any, ty.Any],
                       proc_to_procname_map: ty.Dict[AbstractProcess, str]):
        cache_object["lmt_allocation"] = self._lmt_allocation_dict

        initializers_serializable: ty.List[ty.Tuple[str, str,
                                                    PortInitializer]] = []
        port: AbstractPort
        pi: PortInitializer
        for port, pi in self._initializers_lookup.items():
            procname = proc_to_procname_map[port.process]
            if procname.startswith("Process_"):
                msg = f"Unable to Cache. " \
                      f"Please give unique names to every process. " \
                      f"Violation Name: {procname=}"
                raise Exception(msg)

            initializers_serializable.append((procname, port.name, pi))
        cache_object["initializers"] = initializers_serializable

        cm_serializable: ty.List[ty.Tuple[ty.Tuple[str, str],
                                          ty.Tuple[str, str],
                                          Payload]] = []
        port_pair: PortPair
        payload: Payload
        for port_pair, payload in self.items():
            src_port: AbstractPort = ty.cast(AbstractPort, port_pair.src)
            dst_port: AbstractPort = ty.cast(AbstractPort, port_pair.dst)
            src_proc_name: str = proc_to_procname_map[src_port.process]
            src_port_info = (src_proc_name, src_port.name)
            dst_proc_name: str = proc_to_procname_map[dst_port.process]
            dst_port_info = (dst_proc_name, dst_port.name)
            if src_proc_name.startswith("Process_") or \
                    dst_proc_name.startswith("Process_"):
                msg = f"Unable to Cache. " \
                      f"Please give unique names to every process. " \
                      f"Violation Name: {src_proc_name=} {dst_proc_name=}"
                raise Exception(msg)

            cm_serializable.append((src_port_info, dst_port_info, payload))
        cache_object["channelmap_dict"] = cm_serializable

    def read_from_cache(self,
                        cache_object: ty.Dict[ty.Any, ty.Any],
                        procname_to_proc_map: ty.Dict[str, AbstractProcess]):
        self._lmt_allocation_dict = cache_object["lmt_allocation"]
        initializers_serializable = cache_object["initializers"]
        cm_serializable = cache_object["channelmap_dict"]

        for procname, port_name, pi in initializers_serializable:
            process: AbstractProcess = procname_to_proc_map[procname]
            port: AbstractPort = getattr(process, port_name)
            self._initializers_lookup[port] = pi

        src_port_info: ty.Tuple[str, str]
        dst_port_info: ty.Tuple[str, str]
        payload: Payload
        for src_port_info, dst_port_info, payload in cm_serializable:
            src_port_process: AbstractProcess = procname_to_proc_map[
                src_port_info[0]]
            src: AbstractPort = getattr(src_port_process,
                                        src_port_info[1])
            dst_port_process: AbstractProcess = procname_to_proc_map[
                dst_port_info[0]]
            dst: AbstractPort = getattr(dst_port_process,
                                        dst_port_info[1])
            for port_pair, pld in self.items():
                s, d = port_pair.src, port_pair.dst
                if s.name == src.name and d.name == dst.name and \
                    s.process.name == src_port_process.name and \
                        d.process.name == dst_port_process.name:
                    pld.src_port_initializer = payload.src_port_initializer
                    pld.dst_port_initializer = payload.dst_port_initializer
