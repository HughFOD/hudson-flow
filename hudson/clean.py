"""Clean raw USGS readings and flag anything suspicious.

Philosophy: never silently drop data. Every reading keeps a set of boolean
flag columns explaining what's wrong with it, and `bad` is just the OR of
them. Analysis steps then choose to exclude `bad` rows, and the report can
count exactly how much was excluded and why.
"""
from __future__ import annotations

import polars as pl

from . import config

SERIES = ["site", "param"]
POSITIVE_PARAMS = ("discharge_cfs", "spec_cond_uScm", "do_mgL")


def label(raw: pl.DataFrame) -> pl.DataFrame:
    """Map USGS IDs and parameter codes onto readable names."""
    site_map = {v["id"]: k for k, v in config.SITES.items()}
    return raw.with_columns(
        pl.col("monitoring_location_id").replace_strict(site_map, default=None).alias("site"),
        pl.col("parameter_code").replace_strict(config.PARAMS, default=None).alias("param"),
    ).filter(pl.col("site").is_not_null() & pl.col("param").is_not_null())


def deduplicate(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Snap to the 15-minute grid and resolve duplicate timestamps.

    Returns (deduplicated data, table of conflicting duplicates). Conflicts,
    where the same site/param/time has different values, are worth looking at
    by hand; we keep the most recently modified record.
    """
    df = df.with_columns(pl.col("time").dt.round(config.GRID))
    conflicts = (
        df.group_by(SERIES + ["time"])
        .agg(pl.len().alias("n"), pl.col("value").n_unique().alias("n_values"))
        .filter(pl.col("n_values") > 1)
    )
    deduped = (
        df.sort(SERIES + ["time", "last_modified"])
        .unique(subset=SERIES + ["time"], keep="last", maintain_order=True)
    )
    return deduped, conflicts


def add_flags(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(SERIES + ["time"])
    bad_q = "|".join(config.BAD_QUALIFIERS)
    over = SERIES

    # Flatline: a run of identical values long enough that the sensor is stuck.
    run_id = pl.col("value").rle_id().over(over)
    df = df.with_columns(run_id.alias("_run"))
    df = df.with_columns(pl.len().over(SERIES + ["_run"]).alias("_run_len"))

    # One-point spike: a big jump immediately reversed by a big jump back.
    # Measured on LOG values for flow-like quantities, because sensor noise
    # scales with flow: +2,000 cfs is noise in a flood but a spike at baseflow.
    v = (pl.when(pl.col("param").is_in(POSITIVE_PARAMS))
         .then(pl.col("value").clip(lower_bound=1e-6).log())
         .otherwise(pl.col("value")))
    d = v.diff().over(over)
    df = df.with_columns(d.alias("_d"))
    med = pl.col("_d").median().over(over)
    mad = (pl.col("_d") - med).abs().median().over(over) * 1.4826
    z = ((pl.col("_d") - med) / (mad + 1e-9)).alias("_z")
    df = df.with_columns(z)
    z_next = pl.col("_z").shift(-1).over(over)

    df = df.with_columns(
        pl.col("value").is_null().alias("flag_missing"),
        pl.col("qualifier").str.to_uppercase().str.contains(bad_q).fill_null(False)
        .alias("flag_qualifier"),
        pl.col("qualifier").str.to_uppercase()
        .str.contains("|".join(config.ESTIMATED_QUALIFIERS)).fill_null(False)
        .alias("flag_estimated"),
        (pl.col("param").is_in(POSITIVE_PARAMS) & (pl.col("value") < 0))
        .fill_null(False).alias("flag_negative"),
        (pl.col("_run_len") >= config.FLATLINE_STEPS).alias("flag_flatline"),
        ((pl.col("_z").abs() > config.SPIKE_Z) & (z_next.abs() > config.SPIKE_Z)
         & (pl.col("_z").sign() != z_next.sign())).fill_null(False).alias("flag_spike"),
        (pl.col("approval_status").str.to_lowercase() == "provisional").alias("is_provisional"),
    )
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    return df.with_columns(pl.any_horizontal(flag_cols).alias("bad")).drop(
        ["_run", "_run_len", "_d", "_z"]
    )


def gap_report(df: pl.DataFrame, min_gap: str = "1h") -> pl.DataFrame:
    """Every gap in each series longer than `min_gap`."""
    gaps = (
        df.sort(SERIES + ["time"])
        .with_columns(pl.col("time").diff().over(SERIES).alias("gap"))
        .filter(pl.col("gap") > pl.duration(hours=int(min_gap.rstrip("h"))))
        .select(SERIES + [
            (pl.col("time") - pl.col("gap")).alias("gap_start"),
            pl.col("time").alias("gap_end"),
            pl.col("gap"),
        ])
    )
    return gaps


def flag_summary(df: pl.DataFrame) -> pl.DataFrame:
    flag_cols = [c for c in df.columns if c.startswith("flag_")] + ["is_provisional", "bad"]
    return (
        df.group_by(SERIES)
        .agg([pl.len().alias("n")] + [pl.col(c).sum().alias(c) for c in flag_cols])
        .with_columns((100 * pl.col("bad") / pl.col("n")).round(2).alias("pct_bad"))
        .sort(SERIES)
    )


def clean(raw: pl.DataFrame) -> dict[str, pl.DataFrame]:
    labelled = label(raw)
    deduped, conflicts = deduplicate(labelled)
    flagged = add_flags(deduped)
    return {
        "readings": flagged,
        "conflicts": conflicts,
        "gaps": gap_report(flagged),
        "summary": flag_summary(flagged),
    }


def to_hourly(readings: pl.DataFrame, min_count: int = 2,
              include_bad: bool = False) -> pl.DataFrame:
    """Hourly means of good readings, wide format: one column per site__param.

    An hour needs at least `min_count` good 15-minute readings, otherwise it's
    left null rather than trusting a single reading. `include_bad=True` keeps
    flagged readings, for the anomaly detector, which should see raw data.
    """
    hourly = (
        (readings if include_bad else readings.filter(~pl.col("bad")))
        .with_columns(pl.col("time").dt.truncate("1h"))
        .group_by(SERIES + ["time"])
        .agg(pl.col("value").mean(), pl.len().alias("n"))
        .with_columns(
            pl.when(pl.col("n") >= min_count).then(pl.col("value")).alias("value"),
            (pl.col("site") + "__" + pl.col("param")).alias("col"),
        )
    )
    wide = hourly.pivot(on="col", index="time", values="value").sort("time")
    # Put every hour on the grid so lags are honest even across gaps.
    full = pl.DataFrame({"time": pl.datetime_range(
        wide["time"].min(), wide["time"].max(), "1h", eager=True)})
    return full.join(wide, on="time", how="left")
