# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import AbstractResource


class ResourceMismatchError(Exception):
    def __init__(self,
                 process: AbstractProcess,
                 resources: ty.List[ty.Type[AbstractResource]]) -> None:
        msg = (
            f"The ProcessModel '{type(process.model).__name__}' does not have"
            f"a resource requirement that the subcompiler could handle. The "
            f"subcompiler can handle: "
        )
        for resource in resources:
            msg += f" {resource.__name__}, "
        super().__init__(msg)
