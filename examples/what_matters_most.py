"""Which assumption drives the risk of running out of cash?

Run with:  python examples/what_matters_most.py
"""

from bizmc import PERT, Assumption, Beta, Fixed, Simulation, models
from bizmc.sensitivity import format_tornado, rank_correlations, tornado

sim = Simulation(
    model=models.subscription,
    observed={"starting_cash": 80_000, "starting_customers": 60, "arpa": 120},
    assumptions={
        "churn": Beta.from_counts(9, 300),
        "new_customers": Assumption(PERT(2, 5, 9), vary="period"),
        "cac": PERT(300, 450, 800),
        "gross_margin": PERT(0.70, 0.80, 0.85),
        "fixed_costs": Fixed(9_000),
    },
    periods=36,
)

print("Rank correlation with cash at month 36:")
for row in rank_correlations(sim.run(n=10_000, seed=42), "cash"):
    print(f"  {row.assumption:<14} {row.rho:+.2f}")

print()
print("Tornado: probability of cash < 0 at some point, each assumption at its P10 / P90")
baseline, swings = tornado(
    sim, lambda r: r.probability("cash", below=0, when="any"), n=5_000, seed=42
)
print(format_tornado(baseline, swings))
