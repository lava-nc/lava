# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.proc.lif.process import LogConfig, LearningLIF
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule, \
    Loihi3FLearningRule


class PrototypeLIF(LearningLIF):
    """Prototype Leaky-Integrate-and-Fire (LIF) neural Process with learning
    enabled. Prototype neurons are central piece of the Continually Learning
    Prototypes (CLP) algorithm. """

    pass
