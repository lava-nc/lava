# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from enum import IntEnum


class DownSamplingMethodSparse(IntEnum):
    SKIPPING = 0
    MAX_POOLING = 1


class UpSamplingMethodSparse(IntEnum):
    REPEAT = 0


class DownSamplingMethodDense(IntEnum):
    SKIPPING = 0
    MAX_POOLING = 1
    CONVOLUTION = 2


class UpSamplingMethodDense(IntEnum):
    REPEAT = 0
