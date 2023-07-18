# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC


class AbstractResource(ABC):
    """A hardware resource like a compute resource (e.g., a particular type
    of neuromorphic processor or a CPU), peripheral device, or complete
    system that is required for a ProcessModel.

    Each ProcessModel lists its required hardware resources with the
    @requires decorator.
    """


# Compute resources ------------------------------------------------------------
class AbstractComputeResource(AbstractResource):
    """A compute resource, for example a particular type of neuromorphic
    processor or CPU."""


class CPU(AbstractComputeResource):
    """A central processing unit on a regular computer or laptop."""


class HostCPU(AbstractComputeResource):
    """A central processing unit on a special host system that holds
    neuromorphic devices."""


class GPU(AbstractComputeResource):
    """A graphical processing unit."""


class ECPU(AbstractComputeResource):
    """An embedded central processing unit that is part of a neuromorphic
    chip."""


class LMT(ECPU):
    """A Lakemont embedded central processing unit."""


class PB(ECPU):
    """A Powell Bute embedded central processing unit."""


class NeuroCore(AbstractComputeResource):
    """A neuromorphic core."""


class Loihi1NeuroCore(NeuroCore):
    """A neuromorphic core on a Loihi 1 chip."""


class Loihi2NeuroCore(NeuroCore):
    """A neuromorphic core on a Loihi 2 chip."""


# Peripheral resources ---------------------------------------------------------
class AbstractPeripheralResource(AbstractResource):
    """A hardware resource that is a peripheral device."""


class DVS(AbstractPeripheralResource):
    """An event-based dynamic vision sensor (DVS)."""


class HardDrive(AbstractPeripheralResource):
    """A hard drive in a computer."""


class HeadNodeHardDrive(AbstractPeripheralResource):
    """A hard drive attached to a HeadNode (the node on which a user executes
    code)."""


# Nodes ------------------------------------------------------------------------
class AbstractNode(ABC):
    """A node is a resource that has other compute or peripheral resources."""


class GenericNode(AbstractNode):
    """A generic resource with a regular CPU and a hard drive."""
    resources = [CPU, HardDrive]


class HeadNode(GenericNode):
    """The node on which user executes code, perhaps because processes
    require access to specific disk location.
    """
    resources = [CPU, HeadNodeHardDrive]


class Loihi1System(AbstractNode):
    """A neuromorphic system that carries Loihi 1 chips."""


class KapohoBay(Loihi1System):
    """A KapohoBay system (USB form-factor) that consists of two Loihi 1 chips
    with Lakemont processors."""
    resources = [Loihi1NeuroCore, LMT]


class Nahuku(Loihi1System):
    """A Nahuku system that carries up to 32 Loihi 1 chips."""
    resources = [CPU, Loihi1NeuroCore, LMT]


class Pohoiki(Loihi1System):
    """A system configurable to have one or more Nahuku sub systems."""
    resources = [CPU, Loihi1NeuroCore, LMT]


class Loihi2System(AbstractNode):
    """A neuromorphic system that carries Loihi 2 chips."""


class OheoGulch(Loihi2System):
    """Development and test system with a single Loihi 2 chip."""
    resources = [Loihi2NeuroCore, LMT, PB]


class KapohoPoint(Loihi2System):
    """The smallest form-factor system with four Loihi 2 chips."""
    resources = [Loihi2NeuroCore, LMT, PB]


class Unalaska(Loihi2System):
    resources = [CPU, Loihi2NeuroCore, LMT, PB]
