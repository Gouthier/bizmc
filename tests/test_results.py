import numpy as np
import pytest

from bizmc import SimulationResult


def make(values):
    values = np.asarray(values, dtype=float)
    return SimulationResult(
        outputs={"cash": values},
        observed={},
        assumptions={},
        notes={},
        seed=0,
        n=values.shape[0],
        periods=values.shape[1],
    )


def test_probability_modes():
    r = make([[5, -1, 3], [5, 4, 3], [5, 4, -2], [-1, -1, -1]])
    assert r.probability("cash", below=0) == 0.5  # end
    assert r.probability("cash", below=0, when="any") == 0.75
    assert r.probability("cash", below=0, when=1) == 0.5
    assert r.probability("cash", above=4) == 0.0
    assert r.probability("cash", above=4, when=0) == 0.75
    with pytest.raises(ValueError):
        r.probability("cash")
    with pytest.raises(ValueError):
        r.probability("cash", below=0, above=1)
    with pytest.raises(ValueError):
        r.probability("cash", below=0, when="sometimes")
    with pytest.raises(IndexError):
        r.probability("cash", below=0, when=3)


def test_first_period_below_and_runway():
    r = make([[5, -1, 3], [5, 4, 3], [5, 4, -2], [-1, -1, -1]])
    np.testing.assert_array_equal(r.first_period_below("cash"), [2, np.inf, 3, 1])
    runway = r.runway(q=(0, 100))
    assert runway[0.0] == 1 and runway[100.0] == np.inf


def test_percentiles_mean_at():
    r = make(np.arange(12).reshape(4, 3))
    assert r.percentiles("cash", q=(50,)).shape == (1, 3)
    np.testing.assert_allclose(r.mean("cash"), [4.5, 5.5, 6.5])
    np.testing.assert_array_equal(r.at("cash", 0), [0, 3, 6, 9])
    np.testing.assert_array_equal(r.at("cash"), [2, 5, 8, 11])


def test_unknown_output():
    with pytest.raises(KeyError, match="available"):
        make([[1]])["mrr"]


def test_percentile_table():
    r = make(np.arange(12).reshape(4, 3))
    rows = r.percentile_table(q=(0, 100))
    assert len(rows) == 3
    assert rows[0] == {"period": 1, "output": "cash", "mean": 4.5, "p0": 0.0, "p100": 9.0}
    assert rows[2]["period"] == 3


def test_to_csv_roundtrip(tmp_path):
    import csv

    r = make(np.arange(12).reshape(4, 3))
    path = tmp_path / "out.csv"
    text = r.to_csv(path, q=(50,))
    assert path.read_text(encoding="utf-8") == text
    lines = text.splitlines()
    assert lines[0].startswith("# Projection, not a forecast")
    assert "seed=0" in lines[1]
    data = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert [d["period"] for d in data] == ["1", "2", "3"]
    assert float(data[1]["p50"]) == 5.5


def test_to_dataframe():
    pd = pytest.importorskip("pandas")
    df = make(np.arange(12).reshape(4, 3)).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["period", "output", "mean", "p5", "p50", "p95"]
    assert df.attrs["provenance"]["kind"] == "projection"
