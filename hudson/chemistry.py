"""The chemical engineering checks.

1. Component (salt) balance. Specific conductance is a proxy for dissolved
   salts, which are conserved when two streams mix. So downstream conductance
   should be the flow-weighted average of the inflows:

       C_out = Σ Q_i C_i / Σ Q_i

   This is an independent check on the flow data: if the flow balance closes
   but the salt balance doesn't (or vice versa), one of the sensors is wrong.

2. Dissolved oxygen vs. saturation. Oxygen solubility falls as water warms.
   Comparing measured DO with the saturation value at the measured
   temperature flags readings that are physically implausible.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config
from .balance import col

COND = "spec_cond_uScm"


def do_saturation_mgL(temp_C):
    """Oxygen solubility in fresh water at 1 atm (Benson & Krause, 1984).

    This is the equation used in Standard Methods / by USGS. Input in deg C,
    output in mg/L. Elevation near Troy is low, so the 1 atm assumption is
    fine; add a pressure correction if you move upstream into the mountains.
    """
    T = np.asarray(temp_C, dtype=float) + 273.15
    ln_c = (-139.34411 + 1.575701e5 / T - 6.642308e7 / T**2
            + 1.243800e10 / T**3 - 8.621949e11 / T**4)
    return np.exp(ln_c)


def do_check(hourly: pl.DataFrame) -> pl.DataFrame | None:
    """Percent saturation and plausibility flags for every site with DO + temp."""
    frames = []
    for site in config.SITES:
        do_c, t_c = col(site, "do_mgL"), col(site, "temp_C")
        if do_c not in hourly.columns or t_c not in hourly.columns:
            continue
        df = hourly.select("time", pl.col(do_c).alias("do_mgL"), pl.col(t_c).alias("temp_C"))
        df = df.drop_nulls().with_columns(
            pl.col("temp_C").map_batches(lambda s: pl.Series(do_saturation_mgL(s.to_numpy())))
            .alias("do_sat_mgL")
        ).with_columns(
            (100 * pl.col("do_mgL") / pl.col("do_sat_mgL")).alias("do_sat_pct"),
        ).with_columns(
            (pl.col("do_sat_pct") < config.DO_SAT_LOW_PCT).alias("flag_do_low"),
            (pl.col("do_sat_pct") > config.DO_SAT_HIGH_PCT).alias("flag_do_high"),
            pl.lit(site).alias("site"),
        )
        frames.append(df)
    return pl.concat(frames) if frames else None


def conductance_balance(hourly: pl.DataFrame, lags: dict) -> pl.DataFrame | None:
    """Flow-weighted mixing prediction of downstream conductance."""
    needed = [col(config.OUTFLOW, COND)] + [
        c for s in config.INFLOWS for c in (col(s), col(s, COND))
    ]
    if any(c not in hourly.columns for c in needed):
        return None  # not every site measures conductance; see `run.py inventory`

    q_terms, qc_terms = [], []
    for s in config.INFLOWS:
        L = lags[s]["lag_hours"]
        q = pl.col(col(s)).shift(L)
        c = pl.col(col(s, COND)).shift(L)
        q_terms.append(q)
        qc_terms.append(q * c)

    measured = pl.col(col(config.OUTFLOW, COND))
    df = hourly.select(
        "time",
        measured.alias("cond_measured"),
        pl.sum_horizontal(qc_terms).alias("_qc"),
        pl.sum_horizontal(q_terms).alias("_q"),
        pl.any_horizontal([t.is_null() for t in q_terms + qc_terms]).alias("_incomplete"),
    ).with_columns(
        pl.when(~pl.col("_incomplete")).then(pl.col("_qc") / pl.col("_q")).alias("cond_predicted"),
    ).with_columns(
        (100 * (pl.col("cond_measured") - pl.col("cond_predicted")) / pl.col("cond_measured"))
        .alias("cond_residual_pct"),
    )
    return df.drop(["_qc", "_q", "_incomplete"])
