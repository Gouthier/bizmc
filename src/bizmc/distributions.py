"""Probability distributions used to express *assumptions*.

Every distribution is a small immutable object with a ``sample(rng, size)``
method returning a NumPy array. Parameters are validated at construction time
so that a mistyped assumption fails loudly instead of silently producing
nonsense projections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Distribution",
    "Fixed",
    "Uniform",
    "Normal",
    "LogNormal",
    "Triangular",
    "PERT",
    "Beta",
    "Empirical",
]


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return value


class Distribution:
    """Base class. Subclasses implement :meth:`_draw`."""

    #: Optional ``(low, high)`` bounds applied after drawing (clipping).
    clip: tuple[float | None, float | None] | None = None

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        values = np.asarray(self._draw(rng, size), dtype=float)
        if self.clip is not None:
            low, high = self.clip
            values = np.clip(values, low, high)
        return values

    def _draw(self, rng: np.random.Generator, size):  # pragma: no cover - abstract
        raise NotImplementedError

    def mean(self) -> float:
        """Theoretical mean (before clipping)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(Distribution):
    """A known constant. Use it for values that are not uncertain."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite("value", self.value))

    def _draw(self, rng, size):
        return np.full(size, self.value)

    def mean(self) -> float:
        return self.value


@dataclass(frozen=True)
class Uniform(Distribution):
    """Every value between ``low`` and ``high`` is equally likely."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low, high = _finite("low", self.low), _finite("high", self.high)
        if not low < high:
            raise ValueError(f"Uniform requires low < high, got {low} and {high}")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def _draw(self, rng, size):
        return rng.uniform(self.low, self.high, size)

    def mean(self) -> float:
        return (self.low + self.high) / 2


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian distribution, optionally clipped to ``clip=(low, high)``."""

    mu: float
    sigma: float
    clip: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        mu, sigma = _finite("mu", self.mu), _finite("sigma", self.sigma)
        if sigma <= 0:
            raise ValueError(f"Normal requires sigma > 0, got {sigma}")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    def _draw(self, rng, size):
        return rng.normal(self.mu, self.sigma, size)

    def mean(self) -> float:
        return self.mu


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Positive, right-skewed values, parametrised by their **median** and a
    multiplicative spread ``sigma`` (standard deviation of the log)."""

    median: float
    sigma: float

    def __post_init__(self) -> None:
        median, sigma = _finite("median", self.median), _finite("sigma", self.sigma)
        if median <= 0:
            raise ValueError(f"LogNormal requires median > 0, got {median}")
        if sigma <= 0:
            raise ValueError(f"LogNormal requires sigma > 0, got {sigma}")
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "sigma", sigma)

    def _draw(self, rng, size):
        return rng.lognormal(np.log(self.median), self.sigma, size)

    def mean(self) -> float:
        return float(self.median * np.exp(self.sigma**2 / 2))


@dataclass(frozen=True)
class Triangular(Distribution):
    """Three-point estimate: minimum, most likely, maximum."""

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        low, mode, high = (_finite(n, getattr(self, n)) for n in ("low", "mode", "high"))
        if not (low <= mode <= high and low < high):
            raise ValueError(
                f"Triangular requires low <= mode <= high and low < high, got {low}, {mode}, {high}"
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "high", high)

    def _draw(self, rng, size):
        return rng.triangular(self.low, self.mode, self.high, size)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3


@dataclass(frozen=True)
class PERT(Distribution):
    """Beta-PERT three-point estimate. Smoother than :class:`Triangular` and
    puts less weight on the extremes. ``lamb`` controls the peakedness."""

    low: float
    mode: float
    high: float
    lamb: float = 4.0

    def __post_init__(self) -> None:
        low, mode, high = (_finite(n, getattr(self, n)) for n in ("low", "mode", "high"))
        lamb = _finite("lamb", self.lamb)
        if not (low <= mode <= high and low < high):
            raise ValueError(
                f"PERT requires low <= mode <= high and low < high, got {low}, {mode}, {high}"
            )
        if lamb <= 0:
            raise ValueError(f"PERT requires lamb > 0, got {lamb}")
        for n, v in (("low", low), ("mode", mode), ("high", high), ("lamb", lamb)):
            object.__setattr__(self, n, v)

    def _shape(self) -> tuple[float, float]:
        span = self.high - self.low
        a = 1 + self.lamb * (self.mode - self.low) / span
        b = 1 + self.lamb * (self.high - self.mode) / span
        return a, b

    def _draw(self, rng, size):
        a, b = self._shape()
        return self.low + rng.beta(a, b, size) * (self.high - self.low)

    def mean(self) -> float:
        return (self.low + self.lamb * self.mode + self.high) / (self.lamb + 2)


@dataclass(frozen=True)
class Beta(Distribution):
    """Beta distribution on ``[0, 1]``, convenient for rates (churn, conversion).

    Use :meth:`from_counts` to build it from observed successes and trials,
    which turns real data into an uncertainty-aware assumption.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        a, b = _finite("a", self.a), _finite("b", self.b)
        if a <= 0 or b <= 0:
            raise ValueError(f"Beta requires a > 0 and b > 0, got {a} and {b}")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @classmethod
    def from_counts(cls, successes: int, trials: int, prior: float = 1.0) -> Beta:
        """Posterior of a rate after ``successes`` out of ``trials``
        (uniform ``Beta(prior, prior)`` prior by default)."""
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("need 0 <= successes <= trials")
        return cls(successes + prior, trials - successes + prior)

    def _draw(self, rng, size):
        return rng.beta(self.a, self.b, size)

    def mean(self) -> float:
        return self.a / (self.a + self.b)


@dataclass(frozen=True)
class Empirical(Distribution):
    """Resamples (bootstraps) a set of observed values.

    This is the most direct way to plug *real data* into a simulation: the
    future is assumed to look like the past, nothing more.
    """

    values: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("Empirical requires at least one observed value")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Empirical values must all be finite")
        object.__setattr__(self, "values", tuple(arr.tolist()))

    def _draw(self, rng, size):
        return rng.choice(np.asarray(self.values), size=size, replace=True)

    def mean(self) -> float:
        return float(np.mean(self.values))
