# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.magma.compiler.builders.py_builder import PyProcessBuilder
from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder
from lava.magma.compiler.channel_map import ChannelMap
from lava.magma.compiler.compiler_graphs import ProcGroup
from lava.magma.compiler.subcompilers.channel_builders_factory import (
    ChannelBuildersFactory,
)
from lava.magma.compiler.subcompilers.channel_map_updater import (
    ChannelMapUpdater,
)
from lava.magma.compiler.subcompilers.interfaces import SubCompiler
from lava.magma.compiler.utils import (
    VarInitializer,
    VarPortInitializer,
    PortInitializer,
    LoihiPyInPortInitializer)
from lava.magma.compiler.var_model import PyVarModel, LoihiAddress, \
    LoihiVarModel
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import RefVarTypeMapping, PyVarPort
from lava.magma.core.process.ports.ports import (
    AbstractPort,
    ImplicitVarPort,
    VarPort,
)
from lava.magma.core.process.ports.connection_config import ConnectionConfig
from lava.magma.core.process.process import AbstractProcess
from lava.magma.compiler.subcompilers.constants import SPIKE_BLOCK_CORE

COUNTERS_PER_SPIKE_IO = 65535
SPIKE_IO_COUNTER_START_INDEX = 2

try:
    from lava.magma.core.model.nc.model import AbstractNcProcessModel
except ImportError:
    class AbstractNcProcessModel:
        pass


class _Offset:
    def __init__(self):
        self._offset: int = SPIKE_IO_COUNTER_START_INDEX

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int):
        self._offset = value


class Offset:
    obj: ty.Optional[_Offset] = None

    @staticmethod
    def create() -> _Offset:
        if not Offset.obj:
            Offset.obj = _Offset()
        return Offset.obj

    def get(self) -> int:
        if not Offset.obj:
            return self.create().offset
        else:
            return Offset.obj.offset

    def update(self, value: int):
        if not Offset.obj:
            self.create()
        self.obj.offset = value


class PyProcCompiler(SubCompiler):
    def __init__(
        self,
        proc_group: ProcGroup,
        compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None,
    ):
        """Compiles a group of Processes with ProcessModels that are
        implemented in Python."""
        super().__init__(proc_group, compile_config)
        self._spike_io_counter_offset: Offset = Offset()

    def compile(self, channel_map: ChannelMap,
                partitioning: ty.Dict = None) -> ChannelMap:
        return self._update_channel_map(channel_map)

    def __del__(self):
        Offset.obj = None

    def _update_channel_map(self, channel_map: ChannelMap) -> ChannelMap:
        cm_updater = ChannelMapUpdater(channel_map)
        for process in self._proc_group:
            cm_updater.add_src_ports(process.out_ports)
            cm_updater.add_src_ports(process.ref_ports)
            cm_updater.add_dst_ports(process.in_ports)
            cm_updater.add_dst_ports(process.var_ports)
        return cm_updater.channel_map

    def get_builders(
        self, channel_map: ChannelMap
    ) -> ty.Tuple[ty.Dict[AbstractProcess, AbstractProcessBuilder], ChannelMap]:
        self._tmp_channel_map = channel_map
        builders = {}
        for process in self._proc_group:
            builders[process] = self._create_builder_for_process(process)
        self._tmp_channel_map = None
        return builders, channel_map

    def _create_builder_for_process(
        self, process: AbstractProcess
    ) -> PyProcessBuilder:
        if not issubclass(process.model_class, AbstractPyProcessModel):
            raise TypeError(
                f"ProcessModel of Process '{process.name}' is "
                "incompatible with PyProcCompiler."
            )

        var_initializers = [
            VarInitializer(v.name, v.shape, v.init, v.id) for v in process.vars
        ]
        inport_initializers = self._create_inport_initializers(process)
        outport_initializers = self._create_outport_initializers(process)
        refport_initializers = self._create_refport_initializers(process)
        varport_initializers = self._create_varport_initializers(process)

        process_model_cls = ty.cast(
            ty.Type[AbstractPyProcessModel], process.model_class
        )
        builder = PyProcessBuilder(
            process_model_cls, process.id, process.proc_params
        )
        builder.set_variables(var_initializers)
        builder.set_py_ports(inport_initializers + outport_initializers)
        builder.set_ref_ports(refport_initializers)
        builder.set_var_ports(varport_initializers)

        builder.var_id_to_var_model_map = {
            v.id: PyVarModel(var=v) for v in process.vars
        }

        builder.check_all_vars_and_ports_set()
        return builder

    def _create_inport_initializers(
        self, process: AbstractProcess
    ) -> ty.List[PortInitializer]:
        port_initializers = []
        for port in list(process.in_ports):
            src_ports: ty.List[AbstractPort] = port.get_src_ports()
            is_spike_io_receiver = False
            for src_port in src_ports:
                cls = src_port.process.model_class
                if issubclass(cls, AbstractNcProcessModel):
                    is_spike_io_receiver = True
                elif is_spike_io_receiver:
                    raise Exception("Joining Mixed Processes not Supported")

            if is_spike_io_receiver:
                loihi_addresses = []
                for src_port in src_ports:
                    num_counters = np.prod(src_port.shape)
                    counter_start_idx = self._spike_io_counter_offset.get()

                    loihi_address = LoihiAddress(-1, -1, -1, -1,
                                                 counter_start_idx,
                                                 num_counters,
                                                 1)
                    self._spike_io_counter_offset.update(
                        counter_start_idx + num_counters)
                    loihi_addresses.append(loihi_address)
                loihi_vm = LoihiVarModel(address=loihi_addresses)
                pi = LoihiPyInPortInitializer(
                    port.name,
                    port.shape,
                    ChannelBuildersFactory.get_port_dtype(port),
                    port.__class__.__name__,
                    self._compile_config["pypy_channel_size"],
                    port.get_incoming_transform_funcs(),
                )
                pi.var_model = loihi_vm
                pi.embedded_core = SPIKE_BLOCK_CORE
                pi.embedded_counters = \
                    np.arange(counter_start_idx,
                              counter_start_idx + num_counters, dtype=np.int32)
                if port.connection_configs.values():
                    conn_config = list(port.connection_configs.values())[0]
                else:
                    conn_config = ConnectionConfig()
                pi.connection_config = conn_config
                port_initializers.append(pi)
                self._tmp_channel_map.set_port_initializer(port, pi)
            else:
                pi = PortInitializer(
                    port.name,
                    port.shape,
                    ChannelBuildersFactory.get_port_dtype(port),
                    port.__class__.__name__,
                    self._compile_config["pypy_channel_size"],
                    port.get_incoming_transform_funcs(),
                )
                port_initializers.append(pi)
                self._tmp_channel_map.set_port_initializer(port, pi)
        return port_initializers

    def _create_outport_initializers(
        self, process: AbstractProcess
    ) -> ty.List[PortInitializer]:
        port_initializers = []
        for k, port in enumerate(list(process.out_ports)):
            pi = PortInitializer(
                port.name,
                port.shape,
                ChannelBuildersFactory.get_port_dtype(port),
                port.__class__.__name__,
                self._compile_config["pypy_channel_size"],
                port.get_incoming_transform_funcs(),
            )
            if port.connection_configs.values():
                conn_config = list(port.connection_configs.values())[k]
            else:
                conn_config = ConnectionConfig()
            pi.connection_config = conn_config
            port_initializers.append(pi)
            self._tmp_channel_map.set_port_initializer(port, pi)
        return port_initializers

    def _create_refport_initializers(
        self, process: AbstractProcess
    ) -> ty.List[PortInitializer]:
        port_initializers = []
        for port in list(process.ref_ports):
            pi = PortInitializer(
                port.name,
                port.shape,
                ChannelBuildersFactory.get_port_dtype(port),
                port.__class__.__name__,
                self._compile_config["pypy_channel_size"],
                port.get_outgoing_transform_funcs(),
            )
            port_initializers.append(pi)
            self._tmp_channel_map.set_port_initializer(port, pi)
        return port_initializers

    def _create_varport_initializers(
        self, process: AbstractProcess
    ) -> ty.List[VarPortInitializer]:
        proc_model_cls = process.model_class
        port_initializers = []
        for var_port in list(process.var_ports):
            pi = VarPortInitializer(
                var_port.name,
                var_port.shape,
                var_port.var.name,
                ChannelBuildersFactory.get_port_dtype(var_port),
                var_port.__class__.__name__,
                self._compile_config["pypy_channel_size"],
                self._map_var_port_class(var_port),
                var_port.get_incoming_transform_funcs(),
            )
            port_initializers.append(pi)
            # Set implicit VarPorts (created by connecting a RefPort
            # directly to a Var) as attribute to ProcessModel.
            if isinstance(var_port, ImplicitVarPort):
                setattr(proc_model_cls, var_port.name, var_port)
            self._tmp_channel_map.set_port_initializer(var_port, pi)
        return port_initializers

    @staticmethod
    def _map_var_port_class(port: VarPort) -> ty.Type[PyVarPort]:
        """Maps the port class of a given VarPort from its source RefPort. This
        is needed as implicitly created VarPorts created by connecting RefPorts
        directly to Vars, have no LavaType."""

        # Get the source RefPort of the VarPort
        rp = port.get_src_ports()
        if len(rp) > 0:
            rp = ty.cast(AbstractPort, rp[0])
        else:
            # VarPort is not connect, hence there is no LavaType
            return None

        # Get the LavaType of the RefPort from its ProcessModel
        lt = getattr(rp.process.model_class, rp.name)

        # Return mapping of the RefPort class to VarPort class
        return RefVarTypeMapping.get(lt.cls)
