# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import ABC


class AbstractResource(ABC):
    pass


# Compute resources ------------------------------------------------------------
class AbstractComputeResource(AbstractResource):
    pass


class CPU(AbstractComputeResource):
    pass


class GPU(AbstractComputeResource):
    pass


class ECPU(AbstractComputeResource):
    pass


class LMT(ECPU):
    pass


class PB(ECPU):
    pass


class NeuroCore(AbstractComputeResource):
    pass


class Loihi1NeuroCore(NeuroCore):
    pass


class Loihi2NeuroCore(NeuroCore):
    pass


# Peripheral resources ---------------------------------------------------------
class AbstractPeripheralResource(AbstractResource):
    pass


class DVS(AbstractPeripheralResource):
    pass


class HardDrive(AbstractPeripheralResource):
    pass


class HeadNodeHardDrive(AbstractPeripheralResource):
    pass


# Nodes ------------------------------------------------------------------------
class AbstractNode(ABC):
    """A node is a resource that has other compute or peripheral resources."""

    pass


class GenericNode(AbstractNode):
    resources = [CPU, HardDrive]


class HeadNode(GenericNode):
    """The node on which user executes code, perhaps because processes
    require access to specific disk location.
    Should probably be solved in a different way in the future.
    """

    resources = [CPU, HeadNodeHardDrive]


class Loihi1System(AbstractNode):
    pass


class KapohoBay(Loihi1System):
    resources = [Loihi1NeuroCore, LMT]


class Nahuku(Loihi1System):
    resources = [CPU, Loihi1NeuroCore, LMT]


class Pohoiki(Loihi1System):
    """A system configurable to have one or more Nahuku sub systems."""

    resources = [CPU, Loihi1NeuroCore, LMT]


class Loihi2System(AbstractNode):
    pass


class KapohoPoint(Loihi2System):
    resources = [Loihi2NeuroCore, LMT, PB]


class Unalaska(Loihi2System):
    resources = [CPU, Loihi2NeuroCore, LMT, PB]
