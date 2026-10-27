"""Flow mass balance: upper Hudson + Mohawk should equal Green Island.

    Q_out(t) ≈ Σ Q_in,i(t - lag_i)

Each inflow gets its own travel-time lag, estimated from the data by finding
the shift that best lines up changes in flow (cross-correlation of the
hour-to-hour differences, which ignores the seasonal baseline and focuses on
storm pulses travelling downstream).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config


def col(site: str, param: str = config.FLOW) -> str:
    return f"{site}__{param}"


def estimate_lag(up: np.ndarray, down: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (hours) at which changes upstream best predict changes downstream."""
    du, dd = np.diff(up), np.diff(down)
    best = (0, -np.inf)
    for lag in range(0, max_lag + 1):
        a = du[: len(du) - lag] if lag else du
        b = dd[lag:]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 48:
            continue
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        if r > best[1]:
            best = (lag, float(r))
    return best


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[: len(a) - k] if k else a
    return out


def estimate_lags_joint(hourly: pl.DataFrame, max_lag: int = config.MAX_LAG_HOURS) -> dict:
    """Fit all inflow lags together by minimizing the balance residual.

    Estimating each tributary's lag separately (cross-correlation above) is
    biased when tributary storms are correlated, which they are when the same
    weather system rains on both basins. Searching lag combinations jointly
    against the downstream gauge avoids that.
    """
    from itertools import product

    down = hourly[col(config.OUTFLOW)].to_numpy().astype(float)
    ups = [hourly[col(s)].to_numpy().astype(float) for s in config.INFLOWS]
    best = (None, np.inf)
    for combo in product(range(max_lag + 1), repeat=len(ups)):
        pred = sum(_shift(u, k) for u, k in zip(ups, combo))
        ok = ~(np.isnan(pred) | np.isnan(down)) & (pred > 0)
        if ok.sum() < 48:
            continue
        scale = np.median(down[ok] / pred[ok])  # absorbs ungauged inflow
        err = np.mean(np.abs(down[ok] - scale * pred[ok])) / np.mean(down[ok])
        if err < best[1]:
            best = (combo, err)
    if best[0] is None:
        raise ValueError("Not enough overlapping data to estimate lags")
    out = {}
    for s, u, k in zip(config.INFLOWS, ups, best[0]):
        pair_lag, r = estimate_lag(u, down, max_lag)  # kept for comparison in the report
        out[s] = {"lag_hours": int(k), "pairwise_lag_hours": pair_lag,
                  "pairwise_corr": round(r, 3)}
    out["_fit_error_pct"] = round(100 * best[1], 2)
    return out


def flow_balance(hourly: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    out_col = col(config.OUTFLOW)
    missing = [c for c in [out_col] + [col(s) for s in config.INFLOWS] if c not in hourly.columns]
    if missing:
        raise ValueError(f"Missing discharge columns for the balance: {missing}")

    lag_fit = estimate_lags_joint(hourly)
    lags = {s: lag_fit[s] for s in config.INFLOWS}
    shifted = [pl.col(col(s)).shift(lags[s]["lag_hours"]).alias(f"{s}_lagged")
               for s in config.INFLOWS]

    df = hourly.select(["time", out_col] + [col(s) for s in config.INFLOWS]).with_columns(shifted)
    lagged = [f"{s}_lagged" for s in config.INFLOWS]
    df = df.with_columns(
        # sum_horizontal treats null as 0, so null out rows missing any input
        pl.any_horizontal([pl.col(c).is_null() for c in lagged + [out_col]]).alias("incomplete"),
    ).with_columns(
        pl.when(~pl.any_horizontal([pl.col(c).is_null() for c in lagged]))
        .then(pl.sum_horizontal(lagged)).alias("predicted_cfs"),
    ).with_columns(
        pl.when(~pl.col("incomplete")).then(pl.col(out_col) - pl.col("predicted_cfs"))
        .alias("residual_cfs"),
    ).with_columns(
        (100 * pl.col("residual_cfs") / pl.col(out_col)).alias("residual_pct"),
    )

    # The balance never closes exactly: ungauged creeks add a few percent.
    # So flag departures from the TYPICAL residual, not from zero, otherwise
    # that offset eats into the tolerance and drifts in one direction are
    # caught much later than drifts in the other.
    typical = float(df.filter(~pl.col("incomplete"))["residual_pct"].median() or 0.0)
    over = ((pl.col("residual_pct") - typical).abs() > config.BALANCE_TOL_PCT).cast(pl.Int32)
    df = df.with_columns(
        (over.rolling_sum(config.PERSIST_HOURS) == config.PERSIST_HOURS)
        .fill_null(False).alias("imbalance")
    )

    good = df.filter(~pl.col("incomplete"))
    summary = {
        "typical_residual_pct": round(typical, 2),
        "lags": lags,
        "hours_total": df.height,
        "hours_usable": good.height,
        "median_residual_pct": round(float(good["residual_pct"].median() or np.nan), 2),
        "mean_abs_residual_pct": round(float(good["residual_pct"].abs().mean() or np.nan), 2),
        "pct_hours_imbalanced": round(100 * float(good["imbalance"].mean() or 0), 2),
    }
    return df, summary


def monthly_residuals(balance: pl.DataFrame) -> pl.DataFrame:
    """Median residual by calendar month.

    A seasonal pattern points at something systematic, e.g. water diverted
    into the canal system only during the navigation season, rather than
    random sensor noise. Check the canal's operating dates against this table.
    """
    return (
        balance.filter(~pl.col("incomplete"))
        .group_by(pl.col("time").dt.month().alias("month"))
        .agg(
            pl.col("residual_pct").median().round(2).alias("median_residual_pct"),
            pl.col("residual_cfs").median().round(0).alias("median_residual_cfs"),
            pl.len().alias("hours"),
        )
        .sort("month")
    )
