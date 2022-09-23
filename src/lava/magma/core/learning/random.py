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
    """Trace random generator.

    A TraceRandom generator holds randomly generated numbers for:
    (1) Stochastic rounding after trace decay (float).
    (2) Stochastic rounding after impulse addition (integer).

    A call to the advance method generates new random numbers for each of these.

    Parameters
    ----------
    seed_trace_decay : optional, int
        Seed for random generator of stochastic rounding after trace decay.
    seed_impulse_addition : optional, int
        Seed for random generator of stochastic rounding after impulse addition.
    """

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
            0, 2 ** W_TRACE, size=1, dtype=int
        )[0]

    @property
    def random_trace_decay(self) -> float:
        """Get randomly generated number for stochastic rounding after
        trace decay.

        Returns
        ----------
        random_trace_decay : float
            Randomly generated number for stochastic rounding after trace decay.
        """
        return self._random_trace_decay

    @property
    def random_impulse_addition(self) -> int:
        """Get randomly generated number for stochastic rounding after
        impulse addition.

        Returns
        ----------
        random_trace_decay : int
            Randomly generated number for stochastic rounding after
            impulse addition.
        """
        return self._random_impulse_addition

    def _advance_trace_decay(self) -> None:
        """Generate new random number for stochastic rounding
        after trace decay."""
        self._random_trace_decay = self._rng_trace_decay.random(1)[0]

    def _advance_impulse_addition(self) -> None:
        """Generate new random number for stochastic rounding
        after impulse addition."""
        self._random_impulse_addition = self._rng_impulse_addition.integers(
            0, 2 ** W_TRACE, size=1, dtype=int
        )[0]

    def advance(self) -> None:
        """Generate new random numbers for:
        (1) Stochastic rounding after trace decay.
        (2) Stochastic rounding after impulse addition."""
        self._advance_trace_decay()
        self._advance_impulse_addition()


class ConnVarRandom(AbstractRandomGenerator):
    """Synaptic variable random generator.

    A ConnVarRandom generator holds randomly generated numbers for:
    (1) Stochastic rounding after learning rule application (float).

    A call to the advance method generates new random numbers for each of these.

    Parameters
    ----------
    seed_stochastic_rounding : optional, int
        Seed for random generator of stochastic rounding after learning rule
        application.
    """

    def __init__(self, seed_stochastic_rounding: ty.Optional[int] = 2) -> None:
        self._rng = np.random.default_rng(seed=seed_stochastic_rounding)

        self._random_stochastic_round = self._rng.random(1)[0]

    @property
    def random_stochastic_round(self) -> float:
        """Get randomly generated number for stochastic rounding after
        learning rule application.

        Returns
        ----------
        random_stochastic_round : float
            Randomly generated number for stochastic rounding after
            learning rule application.
        """
        return self._random_stochastic_round

    def _advance_stochastic_round(self) -> None:
        """Generate new random number for stochastic rounding
        after learning rule application."""
        self._random_stochastic_round = self._rng.random(1)[0]

    def advance(self) -> None:
        """Generate new random numbers for:
        (1) Stochastic rounding after learning rule application."""
        self._advance_stochastic_round()
