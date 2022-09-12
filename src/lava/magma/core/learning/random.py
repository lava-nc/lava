# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
from abc import abstractmethod
from lava.magma.core.learning.constants import W_TRACE


class AbstractRandomGenerator:
    """Super class for random generators."""

    @abstractmethod
    def advance(self, *args, **kwargs):
        pass


class TraceRandom(AbstractRandomGenerator):
    """Trace random generator."""

    def __init__(
        self,
        seed_trace_decay: ty.Optional[int] = 0,
        seed_impulse_addition: ty.Optional[int] = 1,
    ) -> None:
        self._rng_trace_decay = np.random.default_rng(
            seed=seed_trace_decay
        )
        self._rng_impulse_addition = np.random.default_rng(
            seed=seed_impulse_addition
        )

        self._random_trace_decay = self._rng_trace_decay.random(1)[0]
        self._random_impulse_addition = self._rng_impulse_addition.integers(
            0, 2**W_TRACE, size=1, dtype=int
        )[0]

    @property
    def random_trace_decay(self) -> float:
        return self._random_trace_decay

    @property
    def random_impulse_addition(self) -> int:
        return self._random_impulse_addition

    def _advance_trace_decay(self) -> None:
        self._random_trace_decay = self._rng_trace_decay.random(1)[0]

    def _advance_impulse_addition(self) -> None:
        self._random_impulse_addition = self._rng_impulse_addition.integers(
            0, 2**W_TRACE, size=1, dtype=int
        )[0]

    def advance(self, *args, **kwargs):
        self._advance_trace_decay()
        self._advance_impulse_addition()


class ConnVarRandom(AbstractRandomGenerator):
    """Synaptic variable random generator."""

    def __init__(self, seed: ty.Optional[int] = 2) -> None:
        self._rng = np.random.default_rng(seed=seed)

        self._random_stochastic_round = self._rng.random(1)[0]

    @property
    def random_stochastic_round(self) -> float:
        return self._random_stochastic_round

    def _advance_stochastic_round(self) -> None:
        self._random_stochastic_round = self._rng.random(1)[0]

    def advance(self):
        self._advance_stochastic_round()
