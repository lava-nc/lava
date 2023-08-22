# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.compiler.builders.channel_builder import (
    ChannelBuilderMp,
    ChannelBuilderNx, ChannelBuilderPyNc,
)
from lava.magma.compiler.channel_map import PortPair, ChannelMap
from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.utils import PortInitializer, LoihiConnectedPortType, \
    LoihiConnectedPortEncodingType
from lava.magma.compiler.var_model import LoihiAddress
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.model import AbstractPyProcessModel

try:
    from lava.magma.core.model.c.model import AbstractCProcessModel
    from lava.magma.core.model.nc.model import AbstractNcProcessModel
except ImportError:
    class AbstractCProcessModel:
        pass

    class AbstractNcProcessModel:
        pass

from lava.magma.core.process.ports.ports import AbstractPort, InPort
from lava.magma.core.process.ports.ports import AbstractSrcPort, AbstractDstPort
from lava.magma.core.process.ports.ports import VarPort, ImplicitVarPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort


class ChannelBuildersFactory:
    """Create ChannelBuilders from a dict-like ChannelMap object with connected
    ports as keys.

    ChannelBuilders allow Runtime to build channels between Process ports.

    The from_channel_map method creates a ChannelBuilder for every connection
    from a source to a destination port in the graph of processes. OutPorts and
    RefPorts are considered source ports while InPorts and VarPorts are
    considered destination ports. A ChannelBuilder is only created for
    terminal connections from one leaf process to another. Intermediate
    ports of a hierarchical process are ignored.

    Once the Runtime has build the channel it can assign the
    corresponding CSP ports to the ProcessBuilder
    (i.e. PyProcBuilder.set_csp_ports(..)) and deploy the Process to the
    appropriate compute node.
    """

    def from_channel_map(
            self,
            channel_map: ChannelMap,
            compile_config: ty.Optional[ty.Dict[str, ty.Any]],
    ) -> ty.List[ChannelBuilderMp]:
        """Create builders for multiprocessing channels between ports in
        connected processes.

        Parameters
        ----------
        channel_map: A dict-like object with a PortPair key for every channel in
            the process network.
        compile_config : ty.Optional[ty.Dict[str, ty.Any]]
            Dictionary that may contain configuration options for the overall
            Compiler as well as all SubCompilers.
        Returns
        -------
        A list of ChannelBuilders with a build method that allows the runtime to
        build the actual channels between connected ports.
        """
        channel_builders = []
        port_pairs = channel_map.keys()
        for port_pair in port_pairs:
            src_port = port_pair.src
            dst_port = port_pair.dst
            initializers = self._get_port_initializers(
                [src_port, dst_port], channel_map
            )
            src_pt_init, dst_pt_init = initializers
            ch_type = self._get_channel_type_from_ports(src_port, dst_port)
            if ch_type is ChannelType.CNc and isinstance(dst_port, VarPort):
                address = self._get_address_for_varport(dst_port)
                src_pt_init.var_model = address
                dst_pt_init.var_model = address
            if ch_type in [ChannelType.CNc, ChannelType.PyNc] and isinstance(
                    dst_port, InPort):
                src_pt_init.var_model = dst_pt_init.var_model
            if ch_type is ChannelType.CNc or ch_type is ChannelType.NcC:
                src_pt_init.connected_port_type = LoihiConnectedPortType.C_NC
                dst_pt_init.connected_port_type = LoihiConnectedPortType.C_NC
                payload = channel_map[port_pair]
                payload.src_port_initializer = src_pt_init
                payload.dst_port_initializer = dst_pt_init

                channel_builder = ChannelBuilderNx(
                    ch_type,
                    src_port.process,
                    dst_port.process,
                    src_pt_init,
                    dst_pt_init,
                )
                channel_builders.append(channel_builder)
            if ch_type in [ChannelType.PyC, ChannelType.CPy]:
                src_pt_init.connected_port_type = LoihiConnectedPortType.C_PY
                dst_pt_init.connected_port_type = LoihiConnectedPortType.C_PY
                if ch_type is ChannelType.PyC:
                    p_port = src_port
                    pi = dst_pt_init
                else:
                    p_port = dst_port
                    pi = src_pt_init
                lt = getattr(p_port.process.model_class, p_port.name).cls
                if lt in [PyInPort.VEC_DENSE, PyOutPort.VEC_DENSE]:
                    pi.connected_port_encoding_type = \
                        LoihiConnectedPortEncodingType.VEC_DENSE
                elif lt in [PyInPort.SCALAR_DENSE, PyOutPort.SCALAR_DENSE]:
                    pi.connected_port_encoding_type = \
                        LoihiConnectedPortEncodingType.SEQ_DENSE
                elif lt in [PyInPort.VEC_SPARSE, PyOutPort.VEC_SPARSE]:
                    pi.connected_port_encoding_type = \
                        LoihiConnectedPortEncodingType.VEC_SPARSE
                else:
                    raise NotImplementedError
            if ch_type in [ChannelType.PyPy, ChannelType.PyC,
                           ChannelType.CPy]:
                channel_builder = ChannelBuilderMp(
                    ch_type,
                    src_port.process,
                    dst_port.process,
                    src_pt_init,
                    dst_pt_init,
                )
                channel_builders.append(channel_builder)
                # Create additional channel builder for every VarPort
                if isinstance(dst_port, VarPort):
                    # RefPort to VarPort connections need channels for
                    # read and write
                    rv_chb = ChannelBuilderMp(
                        ch_type,
                        dst_port.process,
                        src_port.process,
                        dst_pt_init,
                        src_pt_init,
                    )
                    channel_builders.append(rv_chb)

            if ch_type in [ChannelType.PyNc, ChannelType.NcPy]:
                src_pt_init.connected_port_type = LoihiConnectedPortType.PY_NC
                dst_pt_init.connected_port_type = LoihiConnectedPortType.PY_NC

                if ch_type is ChannelType.PyNc:
                    py_port = src_port
                    py_port_init = src_pt_init
                else:
                    py_port = dst_port
                    py_port_init = dst_pt_init

                lt = getattr(py_port.process.model_class, py_port.name).cls
                if lt in [PyInPort.VEC_DENSE, PyOutPort.VEC_DENSE]:
                    py_port_init.connected_port_encoding_type = \
                        LoihiConnectedPortEncodingType.VEC_DENSE
                elif lt in [PyInPort.VEC_SPARSE, PyOutPort.VEC_SPARSE]:
                    py_port_init.connected_port_encoding_type = \
                        LoihiConnectedPortEncodingType.VEC_SPARSE
                else:
                    raise NotImplementedError

                payload = channel_map[port_pair]
                payload.src_port_initializer = src_pt_init
                payload.dst_port_initializer = dst_pt_init

                py_nc_cb = ChannelBuilderPyNc(
                    ch_type,
                    src_port.process,
                    dst_port.process,
                    src_pt_init,
                    dst_pt_init,
                )
                channel_builders.append(py_nc_cb)

        return channel_builders

    @staticmethod
    def _get_port_process_model_class(
            port: AbstractPort,
    ) -> ty.Type[AbstractProcessModel]:
        process = port.process
        return process.model_class

    @staticmethod
    def _get_address_for_varport(varport: VarPort) -> ty.List[LoihiAddress]:
        var_model = varport.var.model
        if var_model:
            return var_model
        return None

    def _get_port_pair_dtypes(
            self, port_pair: PortPair
    ) -> ty.Tuple[ty.Any, ty.Any]:
        for src_port, dst_port in port_pair.src, port_pair.dst:
            src_port_dtype = self.get_port_dtype(src_port)
            dst_port_dtype = self.get_port_dtype(dst_port)
        return src_port_dtype, dst_port_dtype

    @staticmethod
    def get_port_dtype(port: AbstractPort) -> ty.Any:
        """Returns the d_type of a Process Port, as specified in the
        corresponding PortImplementation of the ProcessModel implementing the
        Process"""

        port_pm_class = ChannelBuildersFactory._get_port_process_model_class(
            port
        )
        if hasattr(port_pm_class, port.name):
            if isinstance(port, VarPort):
                return getattr(port_pm_class, port.var.name).d_type
            return getattr(port_pm_class, port.name).d_type
        elif isinstance(port, ImplicitVarPort):
            return getattr(port_pm_class, port.var.name).d_type
        # Port has different name in Process and ProcessModel
        else:
            raise AssertionError(
                "Port {!r} not found in "
                "ProcessModel {!r}".format(port, port_pm_class)
            )

    def _get_channel_type_from_ports(
            self, src_port: AbstractSrcPort, dst_port: AbstractDstPort
    ) -> ChannelType:
        src_pm_class = self._get_port_process_model_class(src_port)
        dst_pm_class = self._get_port_process_model_class(dst_port)
        channel_type = self._get_channel_type_from_processes(
            src_pm_class, dst_pm_class
        )
        return channel_type

    def _get_port_initializers(
            self, ports: ty.List[AbstractPort], channel_map: ChannelMap
    ) -> ty.List[PortInitializer]:
        initializers = []
        for port in ports:
            initializers.append(channel_map.get_port_initializer(port))
        return initializers

    @staticmethod
    def _get_channel_type_from_processes(
            src: ty.Type[AbstractProcessModel],
            dst: ty.Type[AbstractProcessModel]
    ) -> ChannelType:
        """Returns appropriate ChannelType for a given (source, destination)
        pair of ProcessModels."""
        if issubclass(src, AbstractPyProcessModel) and issubclass(
                dst, AbstractPyProcessModel
        ):
            return ChannelType.PyPy
        elif issubclass(src, AbstractPyProcessModel) and issubclass(
                dst, AbstractCProcessModel
        ):
            return ChannelType.PyC
        elif issubclass(src, AbstractCProcessModel) and issubclass(
                dst, AbstractPyProcessModel
        ):
            return ChannelType.CPy
        elif issubclass(src, AbstractCProcessModel) and issubclass(
                dst, AbstractNcProcessModel
        ):
            return ChannelType.CNc
        elif issubclass(src, AbstractNcProcessModel) and issubclass(
                dst, AbstractCProcessModel
        ):
            return ChannelType.NcC
        elif issubclass(src, AbstractNcProcessModel) and issubclass(
                dst, AbstractNcProcessModel
        ):
            return ChannelType.NcNc
        elif issubclass(src, AbstractCProcessModel) and issubclass(
                dst, AbstractCProcessModel
        ):
            return ChannelType.CC
        elif issubclass(src, AbstractPyProcessModel) and issubclass(
                dst, AbstractNcProcessModel
        ):
            return ChannelType.PyNc
        elif issubclass(src, AbstractNcProcessModel) and issubclass(
                dst, AbstractPyProcessModel
        ):
            return ChannelType.NcPy
        else:
            raise NotImplementedError(
                f"No support for (source, destination) pairs of type "
                f"'({src.__name__}, {dst.__name__})' yet."
            )
