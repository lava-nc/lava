# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.magma.compiler.builders.interfaces import ResourceAddress


@dataclass
class NcLogicalAddress(ResourceAddress):
    """
    Represents Logical Id of a resource.
    """
    chip_id: int
    core_id: int


@dataclass
class NcVirtualAddress(ResourceAddress):
    """
    Represents Virtual Id of a resource.
    """
    chip_id: int
    core_id: int
