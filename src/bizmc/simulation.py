"""The simulation engine.

A :class:`Simulation` keeps three things strictly apart:

* **observed** data: facts you measured (current cash, current customers...).
  They are plain numbers and are never randomised.
* **assumptions**: what you believe about the future, each expressed as a
  :class:`~bizmc.distributions.Distribution` wrapped in an :class:`Assumption`.
* **outputs**: what the model computes from the two above. Outputs are
  projections, never facts, and :class:`~bizmc.results.SimulationResult`
  labels them as such.

A name cannot be both observed and assumed: that is checked, so a value you
measured cannot silently be replaced by a guess (or the other way round).
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import numpy as np

from .distributions import Distribution, Fixed
from .results import SimulationResult

__all__ = ["Assumption", "Context", "Simulation"]

Vary = Literal["run", "period"]


@dataclass(frozen=True)
class Assumption:
    """A belief about an uncertain quantity.

    Parameters
    ----------
    distribution:
        The distribution values are drawn from.
    vary:
        ``"run"`` (default): one value per simulated future, constant over
        time. Use it for quantities you are unsure about but expect to be
        stable (for example the true churn rate).
        ``"period"``: a fresh value every period. Use it for quantities that
        fluctuate month to month (for example new sign-ups).
    note:
        Free text explaining where the assumption comes from. It is kept in
        the result's provenance, so the reasoning travels with the numbers.
    """

    distribution: Distribution
    vary: Vary = "run"
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, Distribution):
            raise TypeError("distribution must be a bizmc Distribution")
        if self.vary not in ("run", "period"):
            raise ValueError(f"vary must be 'run' or 'period', got {self.vary!r}")


@dataclass(frozen=True)
class Context:
    """What a model function receives.

    ``draws[name]`` has shape ``(n, 1)`` for ``vary="run"`` assumptions and
    ``(n, periods)`` for ``vary="period"`` ones, so both broadcast naturally
    against ``(n, periods)`` arrays. ``rng`` is a dedicated generator the
    model may use for its own randomness (for example binomial churn).
    """

    observed: Mapping[str, float]
    draws: Mapping[str, np.ndarray]
    n: int
    periods: int
    rng: np.random.Generator

    def period_values(self, name: str, t: int) -> np.ndarray:
        """Values of assumption ``name`` for period ``t``, shape ``(n,)``."""
        arr = self.draws[name]
        return arr[:, 0] if arr.shape[1] == 1 else arr[:, t]


Model = Callable[[Context], Mapping[str, np.ndarray]]


def _stream(seed: int | None, name: str) -> np.random.Generator:
    """An independent random stream per name.

    Deriving each stream from the name (not from the order of declaration)
    means adding, removing or reordering an assumption never changes the
    draws of the others. Two scenarios run with the same seed therefore share
    their random numbers ("common random numbers"), so the difference between
    them comes from the changed assumption, not from sampling noise.
    """
    key = zlib.crc32(name.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))


def _as_assumption(value) -> Assumption:
    if isinstance(value, Assumption):
        return value
    if isinstance(value, Distribution):
        return Assumption(value)
    raise TypeError(
        "assumptions must be Assumption or Distribution objects; "
        "put known numbers in `observed` instead"
    )


@dataclass(frozen=True)
class Simulation:
    """A model plus its observed data and assumptions.

    Example
    -------
    >>> from bizmc import Simulation, Normal, models
    >>> sim = Simulation(
    ...     model=models.cash_runway,
    ...     observed={"starting_cash": 50_000},
    ...     assumptions={
    ...         "revenue": Normal(8_000, 1_500, clip=(0, None)),
    ...         "costs": Normal(11_000, 800, clip=(0, None)),
    ...     },
    ...     periods=24,
    ... )
    >>> result = sim.run(n=2_000, seed=1)
    >>> 0.0 <= result.probability("cash", below=0, when="any") <= 1.0
    True
    """

    model: Model
    observed: Mapping[str, float]
    assumptions: Mapping[str, Assumption | Distribution]
    periods: int
    period_label: str = "month"
    name: str = "simulation"
    _assumptions: Mapping[str, Assumption] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(self.model):
            raise TypeError("model must be callable")
        if int(self.periods) != self.periods or self.periods < 1:
            raise ValueError("periods must be a positive integer")
        observed = {}
        for key, value in dict(self.observed).items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"observed value {key!r} must be finite")
            observed[key] = value
        assumptions = {k: _as_assumption(v) for k, v in dict(self.assumptions).items()}
        overlap = sorted(set(observed) & set(assumptions))
        if overlap:
            raise ValueError(
                f"{overlap} declared both as observed data and as assumptions; "
                "a value is either measured or assumed, not both"
            )
        object.__setattr__(self, "periods", int(self.periods))
        object.__setattr__(self, "observed", MappingProxyType(observed))
        object.__setattr__(self, "_assumptions", MappingProxyType(assumptions))
        object.__setattr__(self, "assumptions", self._assumptions)

    def with_assumptions(self, **changes: Assumption | Distribution | float) -> Simulation:
        """Return a copy with some assumptions replaced (a scenario).

        A plain number is accepted and turned into :class:`Fixed`, which is
        handy for "what if churn were exactly 2 %?" questions.
        """
        merged = dict(self._assumptions)
        for key, value in changes.items():
            if key not in merged:
                raise KeyError(f"unknown assumption {key!r}")
            if isinstance(value, (int, float)):
                value = replace(merged[key], distribution=Fixed(value))
            merged[key] = _as_assumption(value)
        return replace(self, assumptions=merged)

    def with_observed(self, **changes: float) -> Simulation:
        """Return a copy with some observed values updated (new real data)."""
        merged = dict(self.observed)
        for key in changes:
            if key not in merged:
                raise KeyError(f"unknown observed value {key!r}")
        merged.update(changes)
        return replace(self, observed=merged)

    def run(self, n: int = 10_000, seed: int | None = None) -> SimulationResult:
        """Simulate ``n`` possible futures.

        Pass an integer ``seed`` for reproducible results. Without one, a
        fresh seed is drawn and recorded in the result so the run can still
        be reproduced later.
        """
        if int(n) != n or n < 1:
            raise ValueError("n must be a positive integer")
        n = int(n)
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        draws: dict[str, np.ndarray] = {}
        for key, assumption in self._assumptions.items():
            width = 1 if assumption.vary == "run" else self.periods
            values = assumption.distribution.sample(_stream(seed, key), (n, width))
            values.setflags(write=False)
            draws[key] = values
        ctx = Context(
            observed=self.observed,
            draws=MappingProxyType(draws),
            n=n,
            periods=self.periods,
            rng=_stream(seed, "__model__"),
        )
        raw = self.model(ctx)
        outputs: dict[str, np.ndarray] = {}
        for key, value in dict(raw).items():
            arr = np.asarray(value, dtype=float)
            if arr.shape != (n, self.periods):
                raise ValueError(
                    f"model output {key!r} has shape {arr.shape}, expected {(n, self.periods)}"
                )
            outputs[key] = arr
        if not outputs:
            raise ValueError("model returned no outputs")
        return SimulationResult(
            outputs=outputs,
            observed=dict(self.observed),
            assumptions={k: repr(a.distribution) for k, a in self._assumptions.items()},
            notes={k: a.note for k, a in self._assumptions.items() if a.note},
            seed=seed,
            n=n,
            periods=self.periods,
            period_label=self.period_label,
            name=self.name,
            inputs=draws,
        )
