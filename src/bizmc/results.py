"""Simulation results: percentiles, probabilities and provenance.

Everything here describes *simulated* futures. Numbers are projections that
follow from the assumptions, not predictions and certainly not guarantees.
"""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import __version__

__all__ = ["SimulationResult", "DISCLAIMER"]

DISCLAIMER = (
    "Projection, not a forecast: these figures follow from the stated "
    "assumptions and are only as good as they are."
)

When = Literal["end", "any"] | int


@dataclass(frozen=True)
class SimulationResult:
    """Outputs of :meth:`bizmc.Simulation.run`.

    ``outputs[name]`` is an array of shape ``(n, periods)``: one row per
    simulated future, one column per period.

    ``inputs[name]`` holds the values drawn for each assumption: shape
    ``(n, 1)`` for ``vary="run"`` assumptions, ``(n, periods)`` for
    ``vary="period"`` ones. They make it possible to ask *which* assumption
    drives an outcome (see :mod:`bizmc.sensitivity`).
    """

    outputs: Mapping[str, np.ndarray]
    observed: Mapping[str, float]
    assumptions: Mapping[str, str]
    notes: Mapping[str, str]
    seed: int
    n: int
    periods: int
    period_label: str = "month"
    name: str = "simulation"
    inputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    kind: str = field(default="projection", init=False)

    # ------------------------------------------------------------------ access
    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError:
            raise KeyError(f"no output {name!r}; available: {sorted(self.outputs)}") from None

    def _period_index(self, period: int) -> int:
        if not -self.periods <= period < self.periods:
            raise IndexError(f"period {period} out of range for {self.periods} periods")
        return period % self.periods

    # ------------------------------------------------------------- statistics
    def percentiles(self, name: str, q: Sequence[float] = (5, 50, 95)) -> np.ndarray:
        """Percentiles of ``name`` for every period, shape ``(len(q), periods)``."""
        return np.percentile(self[name], q, axis=0)

    def mean(self, name: str) -> np.ndarray:
        """Mean of ``name`` for every period, shape ``(periods,)``."""
        return self[name].mean(axis=0)

    def at(self, name: str, period: int = -1) -> np.ndarray:
        """All simulated values of ``name`` at one period, shape ``(n,)``."""
        return self[name][:, self._period_index(period)]

    def probability(
        self,
        name: str,
        *,
        below: float | None = None,
        above: float | None = None,
        when: When = "end",
    ) -> float:
        """Share of simulated futures in which ``name`` is below/above a threshold.

        ``when="end"`` looks at the last period, ``when="any"`` at whether the
        threshold is crossed at least once, and an integer at that period.
        """
        if (below is None) == (above is None):
            raise ValueError("pass exactly one of below= or above=")
        values = self[name]
        hit = values < below if below is not None else values > above
        if when == "end":
            flags = hit[:, -1]
        elif when == "any":
            flags = hit.any(axis=1)
        elif isinstance(when, (int, np.integer)) and not isinstance(when, bool):
            flags = hit[:, self._period_index(int(when))]
        else:
            raise ValueError("when must be 'end', 'any' or a period index")
        return float(flags.mean())

    def first_period_below(self, name: str, threshold: float = 0.0) -> np.ndarray:
        """For each future, the first period (1-based) where ``name`` drops
        below ``threshold``; ``inf`` if it never does within the horizon.

        Applied to cash, this is the *runway* distribution.
        """
        hit = self[name] < threshold
        first = np.argmax(hit, axis=1).astype(float) + 1
        first[~hit.any(axis=1)] = np.inf
        return first

    def runway(self, name: str = "cash", q: Sequence[float] = (5, 50, 95)) -> dict[float, float]:
        """Percentiles of the runway, in periods. ``inf`` means "beyond the
        simulated horizon", which is all a simulation can honestly say."""
        first = self.first_period_below(name, 0.0)
        # Periods are discrete and may be infinite: no interpolation.
        values = np.percentile(first, q, method="inverted_cdf")
        return {float(p): float(v) for p, v in zip(q, values, strict=True)}

    # ---------------------------------------------------------------- export
    def percentile_table(
        self, names: Sequence[str] | None = None, q: Sequence[float] = (5, 50, 95)
    ) -> list[dict[str, float | int | str]]:
        """Tidy rows ``{"period", "output", "mean", "p5", "p50", ...}``,
        one per period and output: small enough to share or chart anywhere."""
        names = list(names) if names is not None else sorted(self.outputs)
        rows: list[dict[str, float | int | str]] = []
        for name in names:
            pct = self.percentiles(name, q)
            mean = self.mean(name)
            for t in range(self.periods):
                row: dict[str, float | int | str] = {
                    "period": t + 1,
                    "output": name,
                    "mean": float(mean[t]),
                }
                for i, p in enumerate(q):
                    row[f"p{p:g}"] = float(pct[i, t])
                rows.append(row)
        return rows

    def to_csv(
        self,
        path: str | os.PathLike | None = None,
        names: Sequence[str] | None = None,
        q: Sequence[float] = (5, 50, 95),
    ) -> str:
        """Write :meth:`percentile_table` as CSV, preceded by ``#`` comment
        lines carrying the disclaimer, seed and inputs. Returns the CSV text;
        also writes it to ``path`` when given."""
        rows = self.percentile_table(names, q)
        buffer = io.StringIO()
        prov = self.provenance()
        buffer.write(f"# {DISCLAIMER}\n")
        buffer.write(f"# {prov['library']} | {self.name} | n={self.n} | seed={self.seed}\n")
        for key, value in self.observed.items():
            buffer.write(f"# observed {key} = {value:g}\n")
        for key, value in self.assumptions.items():
            buffer.write(f"# assumption {key} ~ {value}\n")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_dataframe(self, names: Sequence[str] | None = None, q: Sequence[float] = (5, 50, 95)):
        """:meth:`percentile_table` as a pandas DataFrame (pandas is optional
        and only imported here). Provenance is stored in ``df.attrs``."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("to_dataframe needs pandas: pip install pandas") from exc
        df = pd.DataFrame(self.percentile_table(names, q))
        df.attrs["provenance"] = self.provenance()
        return df

    # ------------------------------------------------------------- reporting
    def provenance(self) -> dict:
        """Everything needed to understand and reproduce this result."""
        return {
            "kind": self.kind,
            "disclaimer": DISCLAIMER,
            "name": self.name,
            "library": f"bizmc {__version__}",
            "seed": self.seed,
            "n": self.n,
            "periods": self.periods,
            "period_label": self.period_label,
            "observed": dict(self.observed),
            "assumptions": dict(self.assumptions),
            "notes": dict(self.notes),
        }

    def summary(self, names: Sequence[str] | None = None, q: Sequence[float] = (5, 50, 95)) -> str:
        """Plain-text table of end-of-horizon percentiles, with provenance."""
        names = list(names) if names is not None else sorted(self.outputs)
        header = f"{'output':<20}" + "".join(f"{'P' + format(p, 'g'):>14}" for p in q)
        lines = [
            f"{self.name} - {self.n:,} simulated futures, "
            f"{self.periods} {self.period_label}s, seed {self.seed}",
            DISCLAIMER,
            "",
            "Observed data (facts):",
            *(f"  {k} = {v:,g}" for k, v in self.observed.items()),
            "Assumptions (beliefs):",
            *(f"  {k} ~ {v}" for k, v in self.assumptions.items()),
            "",
            f"Projected values at {self.period_label} {self.periods}:",
            header,
        ]
        for name in names:
            values = np.percentile(self.at(name, -1), q)
            lines.append(f"{name:<20}" + "".join(f"{v:>14,.2f}" for v in values))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SimulationResult(name={self.name!r}, n={self.n}, periods={self.periods}, "
            f"seed={self.seed}, outputs={sorted(self.outputs)}, kind='projection')"
        )
