import numpy as np
import pytest

from bizmc import PERT, Beta, Empirical, Fixed, LogNormal, Normal, Triangular, Uniform

RNG_SEED = 1234
N = 200_000


def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "dist",
    [
        Fixed(3.5),
        Uniform(2, 6),
        Normal(10, 2),
        LogNormal(100, 0.4),
        Triangular(1, 2, 6),
        PERT(1, 2, 6),
        Beta(2, 8),
        Empirical([1, 2, 2, 3, 10]),
    ],
)
def test_sample_mean_matches_theory(dist):
    values = dist.sample(rng(), N)
    assert values.shape == (N,)
    assert values.mean() == pytest.approx(dist.mean(), rel=0.01, abs=1e-9)


def test_two_dimensional_shape():
    assert Normal(0, 1).sample(rng(), (7, 3)).shape == (7, 3)


def test_normal_clip():
    values = Normal(0, 1, clip=(0, None)).sample(rng(), 10_000)
    assert values.min() >= 0


def test_ranges_respected():
    assert Uniform(2, 6).sample(rng(), N).min() >= 2
    assert Triangular(1, 2, 6).sample(rng(), N).max() <= 6
    pert = PERT(1, 2, 6).sample(rng(), N)
    assert pert.min() >= 1 and pert.max() <= 6
    beta = Beta(2, 8).sample(rng(), N)
    assert beta.min() >= 0 and beta.max() <= 1


def test_empirical_only_returns_observed_values():
    values = Empirical([4, 8, 15]).sample(rng(), 1000)
    assert set(np.unique(values)) <= {4.0, 8.0, 15.0}


def test_beta_from_counts():
    dist = Beta.from_counts(successes=6, trials=200)
    assert dist.a == 7 and dist.b == 195
    assert dist.mean() == pytest.approx(7 / 202)
    with pytest.raises(ValueError):
        Beta.from_counts(5, 3)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Uniform(5, 5),
        lambda: Normal(0, 0),
        lambda: Normal(float("nan"), 1),
        lambda: LogNormal(0, 1),
        lambda: LogNormal(1, -1),
        lambda: Triangular(1, 7, 6),
        lambda: PERT(3, 3, 3),
        lambda: PERT(1, 2, 3, lamb=0),
        lambda: Beta(0, 1),
        lambda: Empirical([]),
        lambda: Empirical([1, float("inf")]),
        lambda: Fixed(float("inf")),
    ],
)
def test_invalid_parameters_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_distributions_are_immutable():
    dist = Normal(0, 1)
    with pytest.raises(AttributeError):
        dist.mu = 5
