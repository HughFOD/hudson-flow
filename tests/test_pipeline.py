"""Each check must catch the fault planted in the synthetic data."""
import numpy as np
import polars as pl
import pytest

from hudson import balance, chemistry, clean, synthetic


@pytest.fixture(scope="module")
def cleaned():
    return clean.clean(synthetic.make_raw(days=40))


@pytest.fixture(scope="module")
def hourly(cleaned):
    return clean.to_hourly(cleaned["readings"])


def _series(cleaned, site, param):
    return cleaned["readings"].filter((pl.col("site") == site) & (pl.col("param") == param))


def test_flatline_caught(cleaned):
    s = _series(cleaned, "mohawk", "discharge_cfs")
    assert s["flag_flatline"].sum() >= synthetic.FLATLINE[3] - 1


def test_spike_caught(cleaned):
    s = _series(cleaned, "green_island", "discharge_cfs")
    assert s["flag_spike"].sum() >= 1


def test_ice_caught(cleaned):
    s = _series(cleaned, "upper_hudson", "discharge_cfs")
    assert s["flag_qualifier"].sum() == synthetic.ICE[3]


def test_gap_found(cleaned):
    gaps = cleaned["gaps"].filter(pl.col("site") == "green_island")
    assert gaps.height >= 1


def test_conflicting_duplicate_found(cleaned):
    assert cleaned["conflicts"].height == 1


def test_lags_recovered(hourly):
    _, summary = balance.flow_balance(hourly)
    for site, true_lag in synthetic.TRUE_LAGS.items():
        assert abs(summary["lags"][site]["lag_hours"] - true_lag) <= 1


def test_balance_closes_to_ungauged_fraction(hourly):
    _, summary = balance.flow_balance(hourly)
    expected = 100 * synthetic.UNGAUGED_FRACTION / (1 + synthetic.UNGAUGED_FRACTION)
    assert abs(summary["median_residual_pct"] - expected) < 1.0


def test_conductance_balance_closes(hourly):
    _, summary = balance.flow_balance(hourly)
    cb = chemistry.conductance_balance(hourly, summary["lags"])
    assert cb is not None
    assert abs(cb["cond_residual_pct"].drop_nulls().median()) < 2.0


def test_do_saturation_reference_values():
    # Standard table values at 1 atm: ~14.6 mg/L at 0 C, ~9.1 at 20 C
    sat = chemistry.do_saturation_mgL([0.0, 20.0])
    assert sat[0] == pytest.approx(14.62, abs=0.05)
    assert sat[1] == pytest.approx(9.09, abs=0.05)


def test_fouled_do_sensor_flagged(hourly):
    do = chemistry.do_check(hourly)
    assert do["flag_do_high"].sum() > 0
    assert do.head(200)["flag_do_high"].sum() == 0  # healthy period is clean
