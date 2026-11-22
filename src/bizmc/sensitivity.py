"""Which assumption matters most?

Two complementary views:

* :func:`rank_correlations` reuses one simulation run and measures how
  strongly each assumption's drawn values move with an outcome (Spearman rank
  correlation). It is cheap and accounts for all assumptions varying at once.
* :func:`tornado` re-runs the simulation with one assumption pinned at a low
  and then a high quantile, all others left uncertain. It answers "how much
  would the outcome change if this belief turned out low or high?" in the
  outcome's own units. Runs share their random numbers, so the swings reflect
  the assumption, not sampling noise.

Both describe the *model*, not the world: an assumption that looks harmless
here may still be wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .distributions import Fixed
from .results import SimulationResult
from .simulation import Simulation

__all__ = ["Correlation", "Swing", "rank_correlations", "tornado", "format_tornado"]

Statistic = Callable[[SimulationResult], float]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 0, ties sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    # Boundaries of runs of equal values.
    starts = np.flatnonzero(np.r_[True, sorted_vals[1:] != sorted_vals[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return 0.0  # one side is constant: no measurable relationship
    return float((rx * ry).sum() / denom)


@dataclass(frozen=True)
class Correlation:
    """Rank correlation between one assumption and an outcome (-1 to 1)."""

    assumption: str
    rho: float


def rank_correlations(result: SimulationResult, output: str, period: int = -1) -> list[Correlation]:
    """Spearman correlation of each assumption with ``output`` at ``period``.

    Assumptions that vary every period are summarised by their mean over the
    periods up to and including ``period``. Constant (``Fixed``) assumptions
    get 0. Results are sorted by absolute strength, strongest first.
    """
    if not result.inputs:
        raise ValueError("result has no recorded inputs")
    t = result._period_index(period)
    y = result.at(output, t)
    rows = []
    for name, values in result.inputs.items():
        x = values[:, 0] if values.shape[1] == 1 else values[:, : t + 1].mean(axis=1)
        rows.append(Correlation(name, _spearman(x, y)))
    return sorted(rows, key=lambda r: abs(r.rho), reverse=True)


@dataclass(frozen=True)
class Swing:
    """Outcome when one assumption is pinned low, then high."""

    assumption: str
    low_value: float
    high_value: float
    outcome_low: float
    outcome_high: float

    @property
    def swing(self) -> float:
        return abs(self.outcome_high - self.outcome_low)


def _statistic(output: str | Statistic, period: int) -> Statistic:
    if callable(output):
        return output
    return lambda r: float(np.median(r.at(output, period)))


def tornado(
    sim: Simulation,
    output: str | Statistic,
    *,
    n: int = 5_000,
    seed: int = 0,
    q: tuple[float, float] = (10, 90),
    period: int = -1,
    assumptions: Sequence[str] | None = None,
    quantile_samples: int = 100_000,
) -> tuple[float, list[Swing]]:
    """One-at-a-time sensitivity ("tornado") analysis.

    Parameters
    ----------
    output:
        An output name (the statistic is then its median at ``period``) or a
        function ``result -> float``, for example
        ``lambda r: r.probability("cash", below=0, when="any")``.
    q:
        Low and high percentiles each assumption is pinned at.
    assumptions:
        Restrict the analysis to these names (default: all non-fixed ones).

    Returns
    -------
    ``(baseline, swings)``: the statistic with every assumption uncertain,
    and one :class:`Swing` per assumption sorted by size, largest first.
    """
    low_q, high_q = q
    if not 0 <= low_q < high_q <= 100:
        raise ValueError("q must be (low, high) percentiles with 0 <= low < high <= 100")
    stat = _statistic(output, period)
    baseline = stat(sim.run(n=n, seed=seed))
    names = list(assumptions) if assumptions is not None else list(sim.assumptions)
    rng = np.random.default_rng(seed)
    swings = []
    for name in names:
        dist = sim.assumptions[name].distribution  # KeyError for unknown names
        if isinstance(dist, Fixed):
            if assumptions is None:
                continue
            low = high = dist.value
        else:
            low, high = np.percentile(dist.sample(rng, quantile_samples), [low_q, high_q])
        out_low = stat(sim.with_assumptions(**{name: float(low)}).run(n=n, seed=seed))
        out_high = stat(sim.with_assumptions(**{name: float(high)}).run(n=n, seed=seed))
        swings.append(Swing(name, float(low), float(high), out_low, out_high))
    swings.sort(key=lambda s: s.swing, reverse=True)
    return baseline, swings


def format_tornado(baseline: float, swings: Sequence[Swing], width: int = 30) -> str:
    """Plain-text tornado chart, largest swing first."""
    biggest = max((s.swing for s in swings), default=0.0) or 1.0
    lines = [f"baseline: {baseline:,.4g}"]
    label = max((len(s.assumption) for s in swings), default=0)
    for s in swings:
        bar = "#" * max(1, round(width * s.swing / biggest)) if s.swing else ""
        lines.append(
            f"{s.assumption:<{label}}  {bar:<{width}}  "
            f"{s.outcome_low:>12,.4g} .. {s.outcome_high:<12,.4g}"
            f"  (at {s.low_value:,.4g} .. {s.high_value:,.4g})"
        )
    return "\n".join(lines)
