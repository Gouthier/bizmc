import numpy as np
import pytest

from bizmc import PERT, Assumption, Beta, Fixed, Normal, Simulation, models


def subscription_sim(**assumptions):
    base = {
        "churn": Beta.from_counts(successes=9, trials=300),
        "new_customers": Assumption(PERT(2, 5, 9), vary="period"),
        "cac": PERT(300, 450, 800),
        "gross_margin": PERT(0.7, 0.8, 0.85),
        "fixed_costs": Fixed(6_000),
    }
    base.update(assumptions)
    return Simulation(
        model=models.subscription,
        observed={"starting_cash": 80_000, "starting_customers": 60, "arpa": 120},
        assumptions=base,
        periods=36,
    )


def test_subscription_outputs_consistent():
    r = subscription_sim().run(n=2_000, seed=42)
    for key in ("customers", "mrr", "arr", "cash", "net_cash_flow", "churned", "new_customers"):
        assert r[key].shape == (2_000, 36)
    assert r["customers"].min() >= 0
    np.testing.assert_allclose(r["arr"], r["mrr"] * 12)
    # customers(t) = customers(t-1) + new - churned
    prev = np.concatenate([np.full((2_000, 1), 60.0), r["customers"][:, :-1]], axis=1)
    np.testing.assert_allclose(r["customers"], prev + r["new_customers"] - r["churned"])
    np.testing.assert_allclose(np.diff(r["cash"], axis=1), r["net_cash_flow"][:, 1:])


def test_subscription_no_churn_no_growth_is_deterministic():
    r = subscription_sim(
        churn=Fixed(0), new_customers=Fixed(0), gross_margin=Fixed(1), fixed_costs=Fixed(0)
    ).run(n=10, seed=0)
    assert np.all(r["customers"] == 60)
    np.testing.assert_allclose(r["mrr"], 7_200)
    assert r["cash"][0, -1] == pytest.approx(80_000 + 36 * 7_200)


def test_subscription_mean_customers_converge_to_steady_state():
    # Steady state: new / churn = 4 / 0.05 = 80 customers.
    r = (
        subscription_sim(churn=Fixed(0.05), new_customers=Fixed(4))
        .with_observed(starting_customers=80)
        .run(n=4_000, seed=1)
    )
    assert r.mean("customers")[-1] == pytest.approx(80, rel=0.03)


def test_arpa_growth_optional():
    sim = subscription_sim(
        churn=Fixed(0), new_customers=Fixed(0), gross_margin=Fixed(1), fixed_costs=Fixed(0)
    )
    sim = Simulation(
        sim.model, sim.observed, {**sim.assumptions, "arpa_growth": Fixed(0.01)}, sim.periods
    )
    r = sim.run(n=5, seed=0)
    assert r["mrr"][0, -1] == pytest.approx(7_200 * 1.01**36)


def test_cash_runway_vectorised():
    sim = Simulation(
        models.cash_runway,
        {"starting_cash": 10_000},
        {"revenue": Assumption(Normal(1_000, 300), vary="period"), "costs": Fixed(1_500)},
        periods=60,
    )
    r = sim.run(n=5_000, seed=3)
    runway = r.first_period_below("cash")
    # Expected burn 500/month -> about 20 months; noise spreads it.
    assert 15 <= np.median(runway) <= 25


def test_missing_observed_reported():
    sim = Simulation(
        models.subscription,
        {"starting_cash": 1},
        subscription_sim().assumptions,
        periods=3,
    )
    with pytest.raises(KeyError, match="starting_customers"):
        sim.run(n=3, seed=0)
