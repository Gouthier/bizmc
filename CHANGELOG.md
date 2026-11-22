# Changelog

## 0.1.0 - 2026-11-22

First release.

* Distributions: `Fixed`, `Uniform`, `Normal`, `LogNormal`, `Triangular`, `PERT`, `Beta`
  (with `from_counts`), `Empirical`.
* `Simulation` with strict separation of observed data and assumptions, per-name random
  streams, reproducible seeds and scenario helpers (`with_assumptions`, `with_observed`).
* `SimulationResult` with percentiles, probabilities, runway, summary and provenance.
* Built-in models: `cash_runway`, `subscription`.
