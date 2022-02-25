# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from dataclasses import dataclass


@dataclass
class LavaNcType:
    d_type: str
    precision: int = None  # If None, infinite precision is assumed
