# bizmc

**Honest Monte Carlo simulations for small-business and SaaS planning.**

[![CI](https://github.com/Gouthier/bizmc/actions/workflows/ci.yml/badge.svg)](https://github.com/Gouthier/bizmc/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Spreadsheet forecasts give one number per month and hide how unsure that number is.
`bizmc` simulates thousands of possible futures instead, and answers the questions
that actually matter when planning:

* *What is the probability that cash drops below zero in the next 24 months?*
* *How many months of runway do we have, in a bad case and in a typical case?*
* *How much does cutting costs, or lowering churn, change those odds?*

It is a small, dependency-light library (NumPy only), with no network access and no
hidden state.

## Design principles

1. **Facts, beliefs and projections are kept apart.** A `Simulation` takes
   `observed` data (numbers you measured) and `assumptions` (distributions describing
   what you believe). A name cannot be both: the library refuses it, so a guess can
   never silently overwrite a measured value.
2. **Every result is labelled as a projection.** Results carry a disclaimer, the
   observed data, every assumption and its free-text note. The reasoning travels with
   the numbers.
3. **Runs are reproducible.** Pass a `seed` and you get the same result, bit for bit.
   Without one, a seed is drawn *and recorded*, so any run can be replayed.
4. **Scenarios are comparable.** Each assumption gets its own random stream derived
   from its name. Two scenarios run with the same seed share their random numbers, so
   the difference between them comes from what you changed, not from sampling noise.
5. **Real data first.** `Empirical` bootstraps past figures and `Beta.from_counts`
   turns observed counts (for example 9 cancellations in 300 customer-months) into an
   uncertainty-aware rate.

## Installation

```bash
pip install git+https://github.com/Gouthier/bizmc.git
```

Or, to develop:

```bash
git clone https://github.com/Gouthier/bizmc.git
cd bizmc
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from bizmc import PERT, Assumption, Beta, Fixed, Simulation, models

sim = Simulation(
    name="subscription runway",
    model=models.subscription,
    observed={  # facts measured today, never randomised
        "starting_cash": 80_000,
        "starting_customers": 60,
        "arpa": 120,  # average monthly revenue per account
    },
    assumptions={  # beliefs about the future
        "churn": Assumption(Beta.from_counts(9, 300), note="last 12 months of billing"),
        "new_customers": Assumption(PERT(2, 5, 9), vary="period"),
        "cac": PERT(300, 450, 800),
        "gross_margin": PERT(0.70, 0.80, 0.85),
        "fixed_costs": Fixed(9_000),
    },
    periods=36,
)

result = sim.run(n=10_000, seed=42)
print(result.summary(["customers", "mrr", "cash"]))
print(result.probability("cash", below=0, when="any"))  # chance of running out
print(result.runway(q=(10, 50, 90)))  # runway percentiles, in months

leaner = sim.with_assumptions(fixed_costs=7_500).run(n=10_000, seed=42)
print(leaner.probability("cash", below=0, when="any"))
```

Output (NumPy 2.x; exact digits may differ slightly between NumPy versions):

```text
subscription runway - 10,000 simulated futures, 36 months, seed 42
Projection, not a forecast: these figures follow from the stated assumptions and are only as good as they are.

Observed data (facts):
  starting_cash = 80,000
  starting_customers = 60
  arpa = 120
Assumptions (beliefs):
  churn ~ Beta(a=10.0, b=292.0)
  new_customers ~ PERT(low=2.0, mode=5.0, high=9.0, lamb=4.0)
  cac ~ PERT(low=300.0, mode=450.0, high=800.0, lamb=4.0)
  gross_margin ~ PERT(low=0.7, mode=0.8, high=0.85, lamb=4.0)
  fixed_costs ~ Fixed(value=9000.0)

Projected values at month 36:
output                          P5           P50           P95
customers                    89.00        129.00        173.00
mrr                      10,680.00     15,480.00     20,760.00
cash                    -65,375.74     14,807.84     98,914.50

P(cash < 0 at some point in 36 months): 40.9%
Runway percentiles (months): {10.0: 18.0, 50.0: inf, 90.0: inf}
Same question with leaner costs: 8.3%
```

The median business in this example ends with positive cash, yet it runs out of money
at some point in about 4 futures out of 10. That gap is exactly what a single-line
forecast hides. A runway of `inf` means "beyond the simulated horizon", which is all
a simulation can honestly say.

More in [`examples/`](examples/).

## Building blocks

### Distributions

| Class | Use it for |
|---|---|
| `Fixed(value)` | a known constant |
| `Uniform(low, high)` | "anywhere between" |
| `Triangular(low, mode, high)` | quick three-point estimate |
| `PERT(low, mode, high)` | smoother three-point estimate, less weight on extremes |
| `Normal(mu, sigma, clip=(low, high))` | symmetric noise, optionally bounded |
| `LogNormal(median, sigma)` | positive, right-skewed amounts (deal sizes, costs) |
| `Beta(a, b)` / `Beta.from_counts(k, n)` | rates between 0 and 1 (churn, conversion) |
| `Empirical(values)` | resample your own historical data |

All parameters are validated: `Uniform(5, 5)`, `Normal(0, -1)` or `Empirical([])`
raise immediately.

### Assumptions

```python
Assumption(distribution, vary="run", note="")
```

* `vary="run"`: one value per simulated future, constant over time. Use it for things
  you don't know but expect to be stable (the true churn rate).
* `vary="period"`: a new value every period. Use it for things that fluctuate (monthly
  sign-ups). A bare distribution is shorthand for `Assumption(dist)`.

### Built-in models

* `models.cash_runway`: cash from uncertain `revenue` and `costs`.
* `models.subscription`: customers, MRR/ARR, gross profit, acquisition spend and cash for
  a subscription business, with binomial churn and Poisson new customers, so small
  customer bases are as lumpy as they are in reality.

### Your own model

A model is a plain function that receives a `Context` and returns arrays of shape
`(n, periods)`:

```python
import numpy as np
from bizmc import Simulation, Normal, Assumption


def hiring_plan(ctx):
    salaries = ctx.observed["team_size"] * ctx.draws["salary"]  # (n, 1)
    revenue = ctx.draws["revenue"]  # (n, periods)
    cash = ctx.observed["cash"] + np.cumsum(revenue - salaries, axis=1)
    return {"cash": cash}


sim = Simulation(
    model=hiring_plan,
    observed={"cash": 200_000, "team_size": 4},
    assumptions={
        "salary": Normal(5_000, 400),
        "revenue": Assumption(Normal(18_000, 4_000), vary="period"),
    },
    periods=18,
)
```

Draws are read-only, outputs are shape-checked, and missing inputs are reported by name.

### Sensitivity: which assumption matters most?

```python
from bizmc.sensitivity import format_tornado, rank_correlations, tornado

# 1. Cheap: reuse one run, rank assumptions by Spearman correlation with an outcome.
for row in rank_correlations(sim.run(n=10_000, seed=42), "cash"):
    print(f"{row.assumption:<14} {row.rho:+.2f}")

# 2. Tornado: pin each assumption at its P10, then P90, and measure any statistic.
baseline, swings = tornado(
    sim, lambda r: r.probability("cash", below=0, when="any"), n=5_000, seed=42
)
print(format_tornado(baseline, swings))
```

```text
churn          -0.81
cac            -0.33
gross_margin   +0.23
new_customers  +0.11
fixed_costs    +0.00

baseline: 0.4106
churn          ##############################        0.0298 .. 0.9212        (at 0.02075 .. 0.0468)
new_customers  ####################                  0.7816 .. 0.1762        (at 3.432 .. 6.951)
cac            ##############                        0.2272 .. 0.632         (at 367.9 .. 610.1)
gross_margin   #########                              0.551 .. 0.2892        (at 0.7534 .. 0.827)
```

The two views answer different questions. Rank correlations show what explains the
spread *within* one run. The tornado shows what happens if a belief turns out low or
high for the whole horizon: `new_customers` varies month to month, so its random
ups and downs average out within a run (weak correlation), yet being wrong about its
*level* moves the risk a lot. Tornado runs share their random numbers, so swings come
from the assumption, not from sampling noise. Fixed assumptions are skipped.

### Results

| Method | Returns |
|---|---|
| `result["cash"]` | raw array, `(n, periods)` |
| `percentiles(name, q)` | percentiles for every period |
| `mean(name)`, `at(name, period)` | mean path, values at one period |
| `probability(name, below=…/above=…, when="end"\|"any"\|t)` | share of futures meeting the condition |
| `first_period_below(name, threshold)` / `runway(name, q)` | time to cross a threshold |
| `summary()` | plain-text report with facts, beliefs and projections |
| `provenance()` | dict with seed, inputs, notes and library version |
| `percentile_table(names, q)` | tidy rows: period, output, mean, percentiles |
| `to_csv(path, names, q)` | that table as CSV, with the disclaimer and inputs as `#` comment lines |
| `to_dataframe(names, q)` | that table as a pandas DataFrame, provenance in `df.attrs` (needs `pip install "bizmc[pandas]"`) |

## What this library is not

* Not a forecasting oracle: outputs are only as good as the assumptions you give it.
* Not accounting or tax software, and not financial advice.
* Not tuned for any specific industry; the built-in models are deliberately simple and
  meant to be read, copied and adapted.

## Roadmap

* Correlated assumptions (for example acquisition cost rising with volume).
* Optional plotting helpers (fan charts) behind an extra dependency.

Ideas and issues are welcome.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `pip install -e ".[dev]"`, then
`ruff check . && ruff format --check . && pytest`.

## License

[MIT](LICENSE)
