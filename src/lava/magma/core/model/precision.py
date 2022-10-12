# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass


@dataclass
class Precision:
    """Precision information for floating- to fixed-point conversion."""
    is_signed: bool = True
    num_bits: int = None
    implicit_shift: int = None
