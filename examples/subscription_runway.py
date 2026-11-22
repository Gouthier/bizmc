"""How long does the cash last for a small subscription business?

Run with:  python examples/subscription_runway.py
"""

from bizmc import PERT, Assumption, Beta, Fixed, Simulation, models

sim = Simulation(
    name="subscription runway",
    model=models.subscription,
    # Facts measured today. Never randomised.
    observed={
        "starting_cash": 80_000,
        "starting_customers": 60,
        "arpa": 120,  # average monthly revenue per account
    },
    # Beliefs about the future, each with its uncertainty.
    assumptions={
        # 9 of 300 customer-months churned last year -> uncertain rate around 3 %.
        "churn": Assumption(Beta.from_counts(9, 300), note="last 12 months of billing data"),
        # New customers fluctuate month to month.
        "new_customers": Assumption(PERT(2, 5, 9), vary="period", note="sales pipeline estimate"),
        "cac": PERT(300, 450, 800),
        "gross_margin": PERT(0.70, 0.80, 0.85),
        "fixed_costs": Fixed(9_000),
    },
    periods=36,
)

result = sim.run(n=10_000, seed=42)

print(result.summary(["customers", "mrr", "cash"]))
print()
print(
    f"P(cash < 0 at some point in 36 months): {result.probability('cash', below=0, when='any'):.1%}"
)
print(f"Runway percentiles (months): {result.runway(q=(10, 50, 90))}")

# Scenario: what if we cut fixed costs by 1 500 a month?
leaner = sim.with_assumptions(fixed_costs=7_500).run(n=10_000, seed=42)
print(f"Same question with leaner costs: {leaner.probability('cash', below=0, when='any'):.1%}")
