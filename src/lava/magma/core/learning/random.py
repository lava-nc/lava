import typing as ty
import numpy as np
from abc import abstractmethod
from lava.magma.core.learning.constants import BITS_LOW, BITS_HIGH


class AbstractRandomGenerator:
    """Super class for random generators."""

    @abstractmethod
    def advance(self, *args, **kwargs):
        pass


class TraceRandom(AbstractRandomGenerator):
    """Trace random generator."""

    def __init__(
        self,
        seed_rr: ty.Optional[int] = 0,
        seed_rth_trace_decay: ty.Optional[int] = 1,
        seed_impulse_addition: ty.Optional[int] = 2,
    ) -> None:

        self._rr_rng = np.random.default_rng(seed=seed_rr)
        self._rth_trace_decay_rng = np.random.default_rng(
            seed=seed_rth_trace_decay
        )
        self._rth_impulse_addition_rng = np.random.default_rng(
            seed=seed_impulse_addition
        )

        self._rr = self._rr_rng.integers(0, 2**BITS_LOW, size=1)[0]
        self._rth_trace_decay = self._rth_trace_decay_rng.integers(
            0, 2**BITS_LOW, size=1
        )[0]
        self._rth_impulse_addition = self._rth_impulse_addition_rng.integers(
            0, 2**BITS_LOW, size=1
        )[0]

        self.time_bit = 0

    @property
    def rr(self) -> int:
        return self._rr

    @property
    def rth_trace_decay(self) -> int:
        return self._rth_trace_decay

    @property
    def rth_impulse_addition(self) -> int:
        return self._rth_impulse_addition

    def _advance_rr(self) -> None:
        self._rr = self._rr_rng.integers(0, 2**BITS_LOW, size=1)[0]

    def _advance_rth_trace_decay(self) -> None:
        self._rth_trace_decay = self._rth_trace_decay_rng.integers(
            0, 2**BITS_LOW, size=1
        )[0]

    def _advance_rth_impulse_addition(self) -> None:
        self._rth_impulse_addition = self._rth_impulse_addition_rng.integers(
            0, 2**BITS_LOW, size=1
        )[0]

    def advance(self, *args, **kwargs):
        self._advance_rr()
        self._advance_rth_trace_decay()
        self._advance_rth_impulse_addition()


class ConnVarRandom(AbstractRandomGenerator):
    """Synaptic variable random generator."""

    def __init__(self, seed: ty.Optional[int] = None) -> None:
        # self._rng = np.random.default_rng(seed=seed)
        self._rng = np.random.default_rng(seed=3)

        self._rp = self._rng.integers(0, 2 ** (BITS_HIGH - 1), size=1)[0]

    @property
    def rp(self) -> int:
        return self._rp

    def _advance_rp(self) -> None:
        self._rp = self._rng.integers(0, 2 ** (BITS_HIGH - 1), size=1)[0]

    def advance(self):
        self._advance_rp()
