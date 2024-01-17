# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.compiler.builders.interfaces import ResourceAddress
from lava.magma.compiler.channel_map import ChannelMap
from lava.magma.compiler.compiler_utils import split_proc_builders_by_type
from lava.magma.compiler.executable import Executable
from lava.magma.compiler.mappable_interface import Mappable
from lava.magma.compiler.subcompilers.address import (NcLogicalAddress,
                                                      NcVirtualAddress)
from lava.magma.compiler.subcompilers.constants import (NUM_VIRTUAL_CORES_L2,
                                                        NUM_VIRTUAL_CORES_L3)

try:
    from lava.magma.compiler.subcompilers.nc.neurocore. \
        n3_logical_neurocore import \
        N3LogicalNeuroCore
    from lava.magma.core.model.c.ports import CRefPort
except ImportError:
    class N3LogicalNeuroCore:
        pass

    class CRefPort:
        pass

from lava.magma.compiler.var_model import LoihiAddress


class Mapper:
    """
    Assigns virtual addresses to different processes, mappable by mapping
    logical addresses to virtual addresses.
    """
    def __init__(self):
        self.mapper_core_offset: LogicalCoreId = 0
        self.mapper_core_dict: ty.Dict[LogicalCoreId, LogicalCoreId] = {}

    def _set_virtual_address_nc(self, mappable: Mappable, num_cores: int) \
            -> None:
        """
        Sets virtual address for a Neuro Core Mappable.
        Mappable includes : VarPorts, Ports and Vars.

        Parameters
        ----------
        mappable: Mappable to be mapped
        num_cores: Num Cores per NeuroCores

        """
        l_addrs: ty.List[NcLogicalAddress] = mappable.get_logical()
        p_addrs: ty.List[NcVirtualAddress] = []
        for l_addr in l_addrs:
            if l_addr.core_id not in self.mapper_core_dict:
                self.mapper_core_dict[l_addr.core_id] = self.mapper_core_offset
                l_addr.core_id = self.mapper_core_offset
                self.mapper_core_offset += 1
            else:
                l_addr.core_id = self.mapper_core_dict[l_addr.core_id]
            chip_idx = l_addr.core_id // num_cores
            core_idx = l_addr.core_id % num_cores
            p_addrs.append(
                NcVirtualAddress(chip_id=chip_idx, core_id=core_idx))
        mappable.set_virtual(p_addrs)

    def map_cores(self, executable: Executable,
                  channel_map: ChannelMap) -> None:
        """
        This function gets called from the Compiler class once the partition
        is done. It maps logical addresses to virtual addresses.

        Parameters
        ----------
        executable: Compiled Executable

        """
        _, c_builders, nc_builders = split_proc_builders_by_type(
            executable.proc_builders)
        # Iterate over all the ncbuilder and map them
        for ncb in nc_builders.values():
            if isinstance(ncb.compiled_resources[0],
                          N3LogicalNeuroCore):
                num_cores = NUM_VIRTUAL_CORES_L3
            else:
                num_cores = NUM_VIRTUAL_CORES_L2

            p_addrs: ty.List[ResourceAddress] = []
            for resource in ncb.compiled_resources:
                l_addr: ResourceAddress = resource.l_address
                if l_addr.core_id not in self.mapper_core_dict:
                    self.mapper_core_dict[
                        l_addr.core_id] = self.mapper_core_offset
                    l_addr.core_id = self.mapper_core_offset
                    self.mapper_core_offset += 1
                else:
                    l_addr.core_id = self.mapper_core_dict[l_addr.core_id]
                chip_idx = l_addr.core_id // num_cores
                core_idx = l_addr.core_id % num_cores
                p_addrs.append(
                    NcVirtualAddress(chip_id=chip_idx, core_id=core_idx))
            ncb.map_to_virtual(p_addrs)

            for port_initializer in ncb.io_ports.values():
                if port_initializer.var_model is None:
                    continue
                self._set_virtual_address_nc(port_initializer, num_cores)

            for var_model in ncb.var_id_to_var_model_map.values():
                self._set_virtual_address_nc(var_model, num_cores)

            for var_port_initializer in ncb.var_ports.values():
                self._set_virtual_address_nc(var_port_initializer, num_cores)
            self.mapper_core_dict.clear()

        # Iterate over all the cbuilder and map them
        for cb in c_builders.values():
            address: ty.Set[int] = set()
            ports = {**cb.c_ports, **cb.ref_ports}
            # Iterate over all the ports
            for port in ports:
                # Iterate over channel map to find its corresponding
                # src or dst and its initializers
                for port_pair in channel_map:
                    src = port_pair.src
                    # Checking if the initializers are same
                    if channel_map[port_pair].src_port_initializer == ports[
                            port]:
                        var_model = channel_map[
                            port_pair].dst_port_initializer.var_model
                        # Checking to see if its ConvInVarModel or not
                        if hasattr(var_model, "address"):
                            vm = channel_map[
                                port_pair].dst_port_initializer.var_model
                            dst_addr: ty.List[LoihiAddress] = vm.address
                            chips = [addr.physical_chip_id for addr in dst_addr]
                        else:
                            # Will be here for Conv Regions which will have
                            # ConvInVarModel
                            chips = [region.physical_chip_idx for region in
                                     var_model.regions]
                        address.update(chips)
                        # Set address of c ref_var as that of the var port its
                        # pointing to
                        if isinstance(src, CRefPort):
                            payload = channel_map[port_pair]
                            src_initializer = payload.src_port_initializer
                            dst_initializer = payload.dst_port_initializer
                            p_addrs = [
                                NcVirtualAddress(chip_id=addr.p_chip_id,
                                                 core_id=addr.p_core_id)
                                for addr in
                                dst_initializer.address.address]
                            src_initializer.set_virtual(p_addrs)
                        break
                    if channel_map[port_pair].dst_port_initializer == ports[
                            port]:
                        src_addr: ty.List[LoihiAddress] = channel_map[
                            port_pair].src_port_initializer.var_model.address
                        chips = [addr.physical_chip_id for addr in src_addr]
                        address.update(chips)
                        break
            if len(address) > 1 and hasattr(var_model, "address"):
                print('=' * 50)
                print('Note to JOYESH from the future:')
                print('Add logic to make multichip conv input work for YOLO.')
                print('=' * 50)
                raise ValueError("Lava Compiler doesn't support port"
                                 "splitting currently. MultiChip "
                                 "Not Supported ")
            if address:
                cb.address_map.chip_id = address.pop()
