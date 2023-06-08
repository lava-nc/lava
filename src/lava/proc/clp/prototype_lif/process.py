# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.proc.lif.process import LearningLIF


class PrototypeLIF(LearningLIF):
    """Prototype Leaky-Integrate-and-Fire (LIF) neural Process with learning
    enabled. Prototype neurons are central piece of the Continually Learning
    Prototypes (CLP) algorithm. """
