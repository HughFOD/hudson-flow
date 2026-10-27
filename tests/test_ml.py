"""Tests for the forecasting and anomaly-detection layers."""
import numpy as np
import polars as pl
import pytest

from hudson import anomaly, balance, clean, config, features, forecast, synthetic, weather

DAYS = 400


@pytest.fixture(scope="module")
def data():
    raw = synthetic.make_raw(DAYS)
    c = clean.clean(raw)
    hourly = weather.attach_rain(clean.to_hourly(c["readings"]), synthetic.make_rain(DAYS))
    bal, summary = balance.flow_balance(hourly)
    return {"raw": raw, "cleaned": c, "hourly": hourly, "bal": bal, "lags": summary["lags"]}


@pytest.fixture(scope="module")
def result_6h(data):
    return forecast.run_horizon(data["hourly"], data["lags"], 6)


# --- Leakage ------------------------------------------------------------------

def test_features_do_not_use_the_future(data):
    """Scramble everything after a cutoff; features up to the cutoff must not change."""
    h, lags = data["hourly"], data["lags"]
    cut = h.height // 2
    rng = np.random.default_rng(1)
    value_cols = [c for c in h.columns if c != "time"]
    future = h.slice(cut).with_columns(
        [pl.col(c) * pl.Series(rng.uniform(0.5, 2.0, h.height - cut)) for c in value_cols])
    scrambled = pl.concat([h.slice(0, cut), future])

    a = features.build(h, lags, 12).slice(0, cut).drop("y")
    b = features.build(scrambled, lags, 12).slice(0, cut).drop("y")
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            assert np.allclose(x, y, equal_nan=True), f"{c} uses future data"
        else:
            assert (x == y).all(), f"{c} uses future data"


def test_target_is_the_future(data):
    df = features.build(data["hourly"], data["lags"], 6)
    out = data["hourly"][f"{config.OUTFLOW}__{config.FLOW}"]
    assert df["y"][100] == out[106] or (df["y"][100] is None and out[106] is None)


def test_walk_forward_embargo(data):
    times = data["hourly"]["time"]
    for h in (6, 24):
        for train, test in forecast.walk_forward_splits(times, h):
            last_train = times.filter(pl.Series(train)).max()
            first_test = times.filter(pl.Series(test)).min()
            # a training row's target (t + h) must not reach into the test block
            assert (first_test - last_train).total_seconds() >= h * 3600


# --- Forecasting ----------------------------------------------------------------

def test_ml_beats_baselines_at_6h(result_6h):
    t = {r["model"]: r for r in result_6h["table"].iter_rows(named=True)}
    assert t["physics"]["MAE_cfs"] < t["persistence"]["MAE_cfs"]
    assert t["hybrid"]["MAE_cfs"] < t["physics"]["MAE_cfs"]
    assert t["hybrid"]["skill_vs_persistence"] > 0.5


def test_upstream_gauges_matter(result_6h):
    t = {r["model"]: r for r in result_6h["table"].iter_rows(named=True)}
    assert t["gbm_no_upstream"]["MAE_cfs"] > 1.5 * t["gbm"]["MAE_cfs"]


def test_conformal_calibration_helps(result_6h):
    iv = result_6h["interval"]
    gap_raw = abs(iv["coverage_raw_quantile"] - iv["nominal"])
    gap_cal = abs(iv["coverage_conformal"] - iv["nominal"])
    assert gap_cal <= gap_raw + 0.01


def test_physics_baseline_uses_known_upstream(data):
    """At a horizon shorter than the travel time, the physics forecast uses
    an upstream reading from the past, not the latest one."""
    lags, hourly = data["lags"], data["hourly"]
    h = 1  # shorter than both travel times (3 h and 2 h)
    phys = hourly.select(features.physics_forecast_raw(lags, h))["phys_raw"].to_numpy()
    t = 500
    expected = sum(hourly[f"{s}__{config.FLOW}"][t - (lags[s]["lag_hours"] - h)]
                   for s in config.INFLOWS)
    assert phys[t] == pytest.approx(expected)


# --- Anomaly detection ----------------------------------------------------------

def test_drift_invisible_to_sensor_rules_but_caught_by_physics(data):
    import datetime as dt

    c, raw = data["cleaned"], data["raw"]
    ha = clean.to_hourly(c["readings"], include_bad=True)
    det = anomaly.detect(anomaly.build_signals(ha, data["lags"]))
    cmp, _ = anomaly.compare(det, anomaly.rule_flags_hourly(c["readings"], data["bal"]))
    t0 = raw["time"].min()
    a, L = synthetic.DRIFT[2], synthetic.DRIFT[3]
    w = cmp.filter(pl.col("time").is_between(t0 + dt.timedelta(minutes=15 * a),
                                             t0 + dt.timedelta(minutes=15 * (a + L))))
    assert w["rule_sensor"].sum() == 0          # no qualifier, no stuck value: invisible
    assert w["rule_balance"].sum() > 10          # the mass balance sees it
    assert w["ml_anomaly"].sum() > 5             # and so does the model
