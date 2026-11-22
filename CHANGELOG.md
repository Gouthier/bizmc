# Changelog

## Unreleased

* `bizmc.sensitivity`: `rank_correlations` (Spearman, from one run), `tornado`
  (one-at-a-time P10/P90 swings on any statistic, common random numbers) and
  `format_tornado`.
* `SimulationResult.inputs` keeps the values drawn for each assumption.

## 0.1.0 - 2026-11-22

First release.

* Distributions: `Fixed`, `Uniform`, `Normal`, `LogNormal`, `Triangular`, `PERT`, `Beta`
  (with `from_counts`), `Empirical`.
* `Simulation` with strict separation of observed data and assumptions, per-name random
  streams, reproducible seeds and scenario helpers (`with_assumptions`, `with_observed`).
* `SimulationResult` with percentiles, probabilities, runway, summary and provenance.
* Built-in models: `cash_runway`, `subscription`.
