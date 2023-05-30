# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.proc.lif.process import LogConfig, LearningLIF
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule


class PrototypeLIF(LearningLIF):
    """Prototype Leaky-Integrate-and-Fire (LIF) neural Process with learning
    enabled. Prototype neurons are central piece of the Continually Learning
    Prototypes (CLP) algorithm.

        Parameters
        ----------
        shape : tuple(int)
            Number and topology of LIF neurons.
        u : float, list, numpy.ndarray, optional
            Initial value of the neurons' current.
        v : float, list, numpy.ndarray, optional
            Initial value of the neurons' voltage (membrane potential).
        du : float, optional
            Inverse of decay time-constant for current decay. Currently, only a
            single decay can be set for the entire population of neurons.
        dv : float, optional
            Inverse of decay time-constant for voltage decay. Currently, only a
            single decay can be set for the entire population of neurons.
        bias_mant : float, list, numpy.ndarray, optional
            Mantissa part of neuron bias.
        bias_exp : float, list, numpy.ndarray, optional
            Exponent part of neuron bias, if needed. Mostly for fixed point
            implementations. Ignored for floating point implementations.
        vth : float, optional
            Neuron threshold voltage, exceeding which, the neuron will spike.
            Currently, only a single threshold can be set for the entire
            population of neurons.
        log_config: LogConfig, optional
            Configure the amount of debugging output.
        learning_rule: LearningRule
            Defines the learning parameters and equation.
        """

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
            learning_rule: Loihi2FLearningRule = None,
            **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
            learning_rule=learning_rule,
            **kwargs)
