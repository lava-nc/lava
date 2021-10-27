# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from abc import ABC, abstractmethod

from lava.magma.core.model.model import AbstractProcessModel


# ToDo: Move somewhere else. Just created for typing
class AbstractNodeGroup:
    def alloc(self, *args, **kwargs):
        pass


class Net(ABC):
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
    """Abstract interface for a NeuroCore ProcessModels."""

    @abstractmethod
    def allocate(self, net: Net):
        """Allocates resources required by Process via Net provided by
        compiler.
        Note: This should work as before.
        """
        pass
