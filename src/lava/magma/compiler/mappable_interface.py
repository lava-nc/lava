# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing
from abc import ABC

from lava.magma.compiler.builders.interfaces import ResourceAddress


class Mappable(ABC):
    """
    Interface to make entity mappable.
    """

    def set_physical(self, addr: typing.List[ResourceAddress]):
        """
        Parameters
        ----------
        addr : List of PhysicalAddresses to be assigned to the mappable.
        """
        raise NotImplementedError

    def get_logical(self) -> typing.List[ResourceAddress]:
        """
        Returns
        -------
        List of LogicalAddresses.
        """
        raise NotImplementedError
