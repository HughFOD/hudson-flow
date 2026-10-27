"""Leakage-safe features for forecasting Green Island flow h hours ahead.

The rule: a feature stamped at time t may only use data stamped <= t.
Every operation here is a backward shift or a trailing rolling window, and
`tests/test_ml.py` checks it by scrambling the future and confirming the
past features don't change.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config
from .balance import col

FLOW_LAGS = (1, 2, 3, 6, 12, 24)
RAIN_WINDOWS = (3, 6, 12, 24, 48, 72)
META = ["time", "y", "persist", "phys_raw"]


def _flow_features(site: str) -> list[pl.Expr]:
    q = pl.col(col(site))
    name = site
    exprs = [q.alias(f"{name}_q"), q.log().alias(f"{name}_logq")]
    exprs += [q.shift(k).alias(f"{name}_q_lag{k}") for k in FLOW_LAGS]
    exprs += [(q - q.shift(k)).alias(f"{name}_dq{k}") for k in (1, 3, 6, 12)]
    exprs += [
        q.rolling_mean(6, min_samples=3).alias(f"{name}_mean6"),
        q.rolling_mean(24, min_samples=12).alias(f"{name}_mean24"),
        q.rolling_max(24, min_samples=12).alias(f"{name}_max24"),
        (q / q.rolling_mean(24 * 7, min_samples=24)).alias(f"{name}_rel_week"),
    ]
    return exprs


def _rain_features(rain_cols: list[str]) -> list[pl.Expr]:
    exprs = []
    for r in rain_cols:
        for w in RAIN_WINDOWS:
            exprs.append(pl.col(r).rolling_sum(w, min_samples=1).alias(f"{r}_sum{w}"))
    if rain_cols:
        total = pl.sum_horizontal([pl.col(r) for r in rain_cols])
        exprs.append(total.rolling_sum(24, min_samples=1).alias("rain_total_sum24"))
        # Hours since meaningful rain (>1 mm/h in any basin): wetness memory
        wet = (total > 1.0).cast(pl.Int32)
        exprs.append(wet.alias("_wet"))
    return exprs


def physics_forecast_raw(lags: dict, horizon: int) -> pl.Expr:
    """Unscaled mass-balance forecast of outflow at t+h, using only data at <= t.

    Water reaching Green Island at t+h left inflow i at t+h-lag_i. If that
    time is already in the past (h <= lag_i), we know it; otherwise the best
    we can do with the river alone is the latest reading (persistence).
    """
    terms = []
    for s in config.INFLOWS:
        back = lags[s]["lag_hours"] - horizon
        terms.append(pl.col(col(s)).shift(back) if back >= 0 else pl.col(col(s)))
    return pl.sum_horizontal(terms).alias("phys_raw")


def build(hourly: pl.DataFrame, lags: dict, horizon: int) -> pl.DataFrame:
    """One row per hour: META columns plus features. Target y is outflow at t+h."""
    out = pl.col(col(config.OUTFLOW))
    rain_cols = [c for c in hourly.columns if c.startswith("rain_")]
    doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)

    exprs = [pl.col("time"),
             out.shift(-horizon).alias("y"),          # the ONLY forward-looking column
             out.alias("persist"),
             physics_forecast_raw(lags, horizon)]
    for s in config.SITES:
        exprs += _flow_features(s)
    exprs += _rain_features(rain_cols)
    exprs += [
        (2 * np.pi * doy / 365.25).sin().alias("season_sin"),
        (2 * np.pi * doy / 365.25).cos().alias("season_cos"),
    ]
    df = hourly.select(exprs)

    if "_wet" in df.columns:
        # count hours since the last wet hour, resetting at each wet hour
        grp = pl.col("_wet").cum_sum()
        df = df.with_columns(
            pl.int_range(pl.len()).over(grp).cast(pl.Float64).alias("hours_since_rain")
        ).with_columns(
            pl.when(pl.col("_wet").cum_sum() == 0).then(None)
            .otherwise(pl.col("hours_since_rain")).alias("hours_since_rain")
        ).drop("_wet")
    # inflow-to-outflow ratio: how "loaded" the tributaries are right now
    df = df.with_columns((pl.col("phys_raw") / pl.col("persist")).alias("phys_over_persist"))
    return df


def feature_names(df: pl.DataFrame, exclude_groups: tuple[str, ...] = ()) -> list[str]:
    """All feature columns, optionally dropping groups for ablation studies.

    Groups: "rain", "upstream" (upper_hudson / mohawk features), "physics".
    """
    names = [c for c in df.columns if c not in META]
    if "rain" in exclude_groups:
        names = [c for c in names if not (c.startswith("rain_") or c == "hours_since_rain")]
    if "upstream" in exclude_groups:
        names = [c for c in names if not any(c.startswith(s) for s in config.INFLOWS)]
        names = [c for c in names if c != "phys_over_persist"]
    if "physics" in exclude_groups:
        names = [c for c in names if c != "phys_over_persist"]
    return names
