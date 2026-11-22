"""Bootstrap the future from real monthly figures.

The simplest honest model: future months look like past months, drawn at random.
Run with:  python examples/cash_runway_from_history.py
"""

from bizmc import Assumption, Empirical, Simulation, models

# Twelve months of real figures (replace with your own).
revenue_history = [
    7_200,
    6_900,
    8_100,
    7_600,
    9_400,
    8_800,
    7_100,
    9_900,
    10_200,
    8_700,
    9_600,
    11_000,
]
cost_history = [
    9_800,
    10_100,
    9_700,
    10_400,
    10_900,
    10_200,
    9_900,
    11_300,
    10_800,
    10_500,
    11_200,
    11_600,
]

sim = Simulation(
    name="runway from history",
    model=models.cash_runway,
    observed={"starting_cash": 25_000},
    assumptions={
        "revenue": Assumption(Empirical(revenue_history), vary="period", note="last 12 months"),
        "costs": Assumption(Empirical(cost_history), vary="period", note="last 12 months"),
    },
    periods=24,
)

result = sim.run(n=20_000, seed=7)
p5, p50, p95 = result.percentiles("cash", q=(5, 50, 95))[:, 11]
print(f"Cash after 12 months: P5 {p5:,.0f} | median {p50:,.0f} | P95 {p95:,.0f}")
p_out = result.probability("cash", below=0, when="any")
print(f"Probability of running out within 24 months: {p_out:.1%}")
print(f"Runway percentiles (months): {result.runway()}")
