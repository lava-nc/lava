# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC


class AbstractReduceOp(ABC):
    """Reduce operations are required by InPorts to specify how date from
    multiple OutPorts connected to the same InPorts gets integrated."""


class ReduceSum(AbstractReduceOp):
    """ReduceOp to indicate that multiple inputs to same InPort should be
    added."""
