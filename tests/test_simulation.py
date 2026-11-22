import numpy as np
import pytest

from bizmc import DISCLAIMER, Assumption, Fixed, Normal, Simulation, Uniform, models


def runway_sim(**kw):
    params = dict(
        model=models.cash_runway,
        observed={"starting_cash": 30_000},
        assumptions={
            "revenue": Assumption(Normal(5_000, 1_000), vary="period"),
            "costs": Fixed(7_000),
        },
        periods=24,
    )
    params.update(kw)
    return Simulation(**params)


def test_same_seed_same_result():
    a = runway_sim().run(n=500, seed=7)
    b = runway_sim().run(n=500, seed=7)
    np.testing.assert_array_equal(a["cash"], b["cash"])


def test_different_seed_different_result():
    a = runway_sim().run(n=500, seed=7)
    b = runway_sim().run(n=500, seed=8)
    assert not np.array_equal(a["cash"], b["cash"])


def test_seed_recorded_when_not_given():
    sim = runway_sim()
    result = sim.run(n=200)
    again = sim.run(n=200, seed=result.seed)
    np.testing.assert_array_equal(result["cash"], again["cash"])


def test_adding_an_assumption_keeps_other_draws():
    """Streams are keyed by name, so unrelated assumptions don't interfere."""

    def model(ctx):
        return {"revenue": np.broadcast_to(ctx.draws["revenue"], (ctx.n, ctx.periods))}

    base = dict(model=model, observed={}, periods=3)
    one = Simulation(assumptions={"revenue": Normal(0, 1)}, **base).run(n=100, seed=3)
    two = Simulation(assumptions={"aaa": Uniform(0, 1), "revenue": Normal(0, 1)}, **base).run(
        n=100, seed=3
    )
    np.testing.assert_array_equal(one["revenue"], two["revenue"])


def test_deterministic_inputs_give_exact_cash():
    sim = runway_sim(assumptions={"revenue": Fixed(5_000), "costs": Fixed(7_000)}, periods=3)
    result = sim.run(n=10, seed=0)
    np.testing.assert_allclose(result["cash"][0], [28_000, 26_000, 24_000])
    assert result.probability("cash", below=0, when="any") == 0.0


def test_vary_run_is_constant_over_time():
    def model(ctx):
        return {"x": np.broadcast_to(ctx.draws["x"], (ctx.n, ctx.periods))}

    result = Simulation(model, {}, {"x": Normal(0, 1)}, periods=5).run(n=50, seed=1)
    assert np.all(result["x"] == result["x"][:, :1])
    result = Simulation(model, {}, {"x": Assumption(Normal(0, 1), vary="period")}, periods=5).run(
        n=50, seed=1
    )
    assert not np.all(result["x"] == result["x"][:, :1])


def test_observed_and_assumed_cannot_overlap():
    with pytest.raises(ValueError, match="both"):
        runway_sim(
            observed={"starting_cash": 1, "revenue": 5_000},
        )


def test_plain_numbers_are_not_assumptions():
    with pytest.raises(TypeError, match="observed"):
        runway_sim(assumptions={"revenue": 5_000, "costs": Fixed(1)})


def test_invalid_inputs():
    with pytest.raises(ValueError):
        runway_sim(periods=0)
    with pytest.raises(ValueError):
        runway_sim(observed={"starting_cash": float("nan")})
    with pytest.raises(ValueError):
        runway_sim().run(n=0)
    with pytest.raises(ValueError):
        Assumption(Fixed(1), vary="week")
    with pytest.raises(TypeError):
        Assumption(3)


def test_missing_inputs_reported():
    sim = runway_sim(assumptions={"revenue": Fixed(1)})
    with pytest.raises(KeyError, match="costs"):
        sim.run(n=5, seed=0)


def test_bad_model_output_shape():
    sim = Simulation(lambda ctx: {"x": np.zeros(ctx.n)}, {}, {}, periods=4)
    with pytest.raises(ValueError, match="shape"):
        sim.run(n=5, seed=0)


def test_model_cannot_mutate_draws():
    def model(ctx):
        ctx.draws["x"][0, 0] = 99
        return {}

    with pytest.raises(ValueError):
        Simulation(model, {}, {"x": Fixed(1)}, periods=2).run(n=3, seed=0)


def test_scenarios_share_random_numbers():
    base = runway_sim()
    cheaper = base.with_assumptions(costs=6_000)
    a, b = base.run(n=300, seed=11), cheaper.run(n=300, seed=11)
    # Only costs changed, by exactly 1 000 per month.
    np.testing.assert_allclose(
        b["cash"] - a["cash"], np.broadcast_to(np.arange(1, 25) * 1_000.0, (300, 24))
    )
    assert base.assumptions["costs"].distribution == Fixed(7_000)


def test_with_observed_and_unknown_keys():
    sim = runway_sim().with_observed(starting_cash=0)
    assert sim.observed["starting_cash"] == 0
    with pytest.raises(KeyError):
        runway_sim().with_observed(nope=1)
    with pytest.raises(KeyError):
        runway_sim().with_assumptions(nope=1)


def test_provenance_and_labelling():
    sim = runway_sim(
        assumptions={
            "revenue": Assumption(Normal(5_000, 1_000), vary="period", note="last 6 months"),
            "costs": Fixed(7_000),
        }
    )
    result = sim.run(n=100, seed=5)
    prov = result.provenance()
    assert prov["kind"] == "projection"
    assert prov["seed"] == 5
    assert prov["observed"] == {"starting_cash": 30_000}
    assert prov["notes"] == {"revenue": "last 6 months"}
    assert "Normal" in prov["assumptions"]["revenue"]
    text = result.summary()
    assert DISCLAIMER in text and "Observed data" in text and "Assumptions" in text
    assert "projection" in repr(result)


def test_result_keeps_drawn_inputs():
    result = runway_sim().run(n=50, seed=2)
    assert set(result.inputs) == {"revenue", "costs"}
    assert result.inputs["revenue"].shape == (50, 24)  # vary="period"
    assert result.inputs["costs"].shape == (50, 1)  # vary="run"
    np.testing.assert_allclose(
        result["net_cash_flow"], result.inputs["revenue"] - result.inputs["costs"]
    )
