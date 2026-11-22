import numpy as np
import pytest

from bizmc import Assumption, Fixed, Normal, Simulation, SimulationResult, Uniform, models
from bizmc.sensitivity import (
    _average_ranks,
    _spearman,
    format_tornado,
    rank_correlations,
    tornado,
)


def runway():
    return Simulation(
        models.cash_runway,
        {"starting_cash": 50_000},
        {
            # costs are far more uncertain than revenue
            "revenue": Normal(10_000, 200),
            "costs": Uniform(8_000, 14_000),
            "unused": Fixed(1),
        },
        periods=12,
    )


def test_average_ranks_handles_ties():
    np.testing.assert_array_equal(_average_ranks(np.array([10, 20, 20, 5])), [1, 2.5, 2.5, 0])


def test_spearman_basics():
    x = np.arange(10.0)
    assert _spearman(x, x**3) == pytest.approx(1)
    assert _spearman(x, -x) == pytest.approx(-1)
    assert _spearman(x, np.ones(10)) == 0.0


def test_rank_correlations_order_and_sign():
    result = runway().run(n=4_000, seed=1)
    rows = rank_correlations(result, "cash")
    assert [r.assumption for r in rows] == ["costs", "revenue", "unused"]
    assert rows[0].rho < -0.9  # higher costs -> less cash
    assert rows[1].rho > 0
    assert rows[2].rho == 0.0


def test_rank_correlations_period_varying_inputs():
    sim = Simulation(
        models.cash_runway,
        {"starting_cash": 0},
        {"revenue": Assumption(Normal(100, 30), vary="period"), "costs": Fixed(90)},
        periods=6,
    )
    rows = rank_correlations(sim.run(n=2_000, seed=0), "cash", period=2)
    assert rows[0].assumption == "revenue" and rows[0].rho > 0.95


def test_rank_correlations_requires_inputs():
    bare = SimulationResult({"x": np.zeros((2, 1))}, {}, {}, {}, 0, 2, 1)
    with pytest.raises(ValueError):
        rank_correlations(bare, "x")


def test_tornado_ranks_and_units():
    baseline, swings = tornado(runway(), "cash", n=2_000, seed=3)
    assert [s.assumption for s in swings] == ["costs", "revenue"]  # Fixed skipped
    costs = swings[0]
    # Costs pinned at P10 / P90 of Uniform(8k, 14k): 8 600 and 13 400.
    assert costs.low_value == pytest.approx(8_600, rel=0.01)
    assert costs.high_value == pytest.approx(13_400, rel=0.01)
    # 12 months x 4 800 difference in monthly costs.
    assert costs.outcome_low - costs.outcome_high == pytest.approx(12 * 4_800, rel=0.02)
    assert costs.outcome_high < baseline < costs.outcome_low


def test_tornado_custom_statistic_and_selection():
    def p_short(r):
        return r.probability("cash", below=0, when="any")

    baseline, swings = tornado(runway(), p_short, n=1_000, seed=0, assumptions=["costs", "unused"])
    assert 0 <= baseline <= 1
    assert {s.assumption for s in swings} == {"costs", "unused"}
    unused = next(s for s in swings if s.assumption == "unused")
    assert unused.swing == 0


def test_tornado_is_reproducible_and_validates():
    a = tornado(runway(), "cash", n=500, seed=9)
    b = tornado(runway(), "cash", n=500, seed=9)
    assert a == b
    with pytest.raises(ValueError):
        tornado(runway(), "cash", q=(90, 10))
    with pytest.raises(KeyError):
        tornado(runway(), "cash", assumptions=["nope"])


def test_format_tornado():
    baseline, swings = tornado(runway(), "cash", n=500, seed=0)
    text = format_tornado(baseline, swings)
    lines = text.splitlines()
    assert lines[0].startswith("baseline")
    assert lines[1].startswith("costs") and "#" in lines[1]
    assert format_tornado(1.0, []) == "baseline: 1"
