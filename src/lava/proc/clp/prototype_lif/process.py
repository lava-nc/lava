# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.proc.lif.process import LearningLIF
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule, \
    Loihi3FLearningRule
from lava.magma.core.process.process import LogConfig
from lava.magma.core.process.ports.ports import InPort


class PrototypeLIF(LearningLIF):
    """Prototype Leaky-Integrate-and-Fire (LIF) neural Process with learning
    enabled. Prototype neurons are central piece of the Continually Learning
    Prototypes (CLP) algorithm. """

    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            du: ty.Optional[float] = 0,
            dv: ty.Optional[float] = 0,
            bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            vth: ty.Optional[float] = 10,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
            learning_rule: ty.Union[
                Loihi2FLearningRule, Loihi3FLearningRule] = None,
            **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            vth=vth,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            learning_rule=learning_rule,
            **kwargs,
        )

        self.reset_in = InPort(shape=shape)
