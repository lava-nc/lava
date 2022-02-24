# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from abc import ABC, abstractmethod
import logging
import typing as ty

from lava.magma.core.model.interfaces import AbstractNodeGroup
from lava.magma.core.model.model import AbstractProcessModel


class Net(ABC):
    """Represents a collection of logical entities (Attribute Groups)
    that consume resources on a NeuroCore.

    * InputAxons
    * Synapses
    * DendriticAccumulator
    * Compartments
    * OutputAxons
    * Synaptic pre traces
    * Synaptic post traces
    """
    def __init__(self):
        self.out_ax = AbstractNodeGroup()
        self.cx = AbstractNodeGroup()
        self.cx_profile_cfg = AbstractNodeGroup()
        self.vth_profile_cfg = AbstractNodeGroup()
        self.cx_cfg = AbstractNodeGroup()
        self.da = AbstractNodeGroup()
        self.da_cfg = AbstractNodeGroup()
        self.syn = AbstractNodeGroup()
        self.syn_cfg = AbstractNodeGroup()
        self.in_ax = AbstractNodeGroup()

    def connect(self, from_thing, to_thing):
        pass


class AbstractNcProcessModel(AbstractProcessModel, ABC):
    """Abstract interface for NeuroCore ProcessModels

    Example for how variables might be initialized:
        u: np.ndarray =    LavaNcType(np.ndarray, np.int32, precision=24)
        v: np.ndarray =    LavaNcType(np.ndarray, np.int32, precision=24)
        bias: np.ndarray = LavaNcType(np.ndarray, np.int16, precision=12)
        du: int =          LavaNcType(int, np.uint16, precision=12)
    """
    def __init__(self,
                 log: logging.getLoggerClass,
                 proc_params: ty.Dict[str, ty.Any],
                 loglevel: int = logging.WARNING) -> None:
        super().__init__(proc_params, loglevel=loglevel)
        self.model_id: ty.Optional[int] = None

    @abstractmethod
    def allocate(self, net: Net):
        """Allocates resources required by Process via Net provided by
        compiler.
        """
        pass


class NcProcessModel(AbstractNcProcessModel):
    def __init__(self,
                 proc_params: ty.Dict[str, ty.Any],
                 loglevel: int = logging.WARNING):
        super(AbstractNcProcessModel, self).__init__(proc_params,
                                                     loglevel=loglevel)

    def allocate(self, net: Net):
        pass

    def run(self):
        pass
