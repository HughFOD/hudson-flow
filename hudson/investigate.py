"""Why doesn't the flow balance close? Three tests.

1. Canal test. During the navigation season, part of the Mohawk is diverted
   around the Cohoes gauge through the Waterford Flight of locks and enters
   the Hudson between the upstream gauges and Green Island (NYPA's Crescent
   relicensing documents note that Cohoes flow statistics exclude it). If
   that diversion explained the gap, the gap in cfs would step up when the
   canal opens and down when it closes.

2. Month-by-month gap in cfs. A gap that appears and disappears abruptly,
   unrelated to season or flow, points at a measurement problem rather than
   anything physical in the river.

3. Calibration check. USGS crews measure each river by hand several times a
   year; the gauge's continuous record is built from a rating curve fitted
   to those visits. Comparing each hand measurement with what the gauge
   reported at the same moment shows which gauge drifted, and when.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .balance import col

# NYS Canal Corporation navigation seasons (opening 7 am, closing evening).
# Sources: canals.ny.gov notices and announcements for each year.
CANAL_SEASONS = {
    2022: ("2022-05-20", "2022-10-12"),
    2023: ("2023-05-19", "2023-10-11"),
    2024: ("2024-05-17", "2024-10-16"),
    2025: ("2025-05-16", "2025-11-03"),
}


def daily_gap(balance: pl.DataFrame) -> pl.DataFrame:
    out = col(config.OUTFLOW)
    d = (balance.filter(~pl.col("incomplete"))
         .group_by(pl.col("time").dt.date().alias("date"))
         .agg(pl.col(out).mean().alias("outflow_cfs"), pl.col("predicted_cfs").mean().alias("inflow_cfs"),
              pl.len().alias("hours"))
         .filter(pl.col("hours") >= 20)
         .with_columns((pl.col("outflow_cfs") - pl.col("inflow_cfs")).alias("gap_cfs"))
         .with_columns((100 * pl.col("gap_cfs") / pl.col("outflow_cfs")).alias("gap_pct"))
         .sort("date"))

    def in_season(x: date) -> bool:
        s = CANAL_SEASONS.get(x.year)
        return bool(s) and date.fromisoformat(s[0]) <= x <= date.fromisoformat(s[1])

    return d.with_columns(pl.col("date").map_elements(in_season, return_dtype=pl.Boolean).alias("canal_open"))


def canal_test(daily: pl.DataFrame) -> dict:
    by_state = (daily.group_by("canal_open").agg(
        pl.len().alias("days"), pl.col("gap_cfs").median().alias("median_gap_cfs"),
        pl.col("gap_pct").median().alias("median_gap_pct"),
        pl.col("outflow_cfs").median().alias("median_outflow_cfs")).sort("canal_open"))
    events = []
    for y, (o, c) in CANAL_SEASONS.items():
        for kind, day in (("opens", o), ("closes", c)):
            D = date.fromisoformat(day)
            before = daily.filter(pl.col("date").is_between(D - timedelta(days=14), D - timedelta(days=1)))
            after = daily.filter(pl.col("date").is_between(D + timedelta(days=1), D + timedelta(days=14)))
            if before.height >= 7 and after.height >= 7:
                events.append({"year": y, "event": kind, "date": day,
                               "gap_before_cfs": float(before["gap_cfs"].median()),
                               "gap_after_cfs": float(after["gap_cfs"].median())})
    ev = pl.DataFrame(events) if events else pl.DataFrame()
    if ev.height:
        ev = ev.with_columns((pl.col("gap_after_cfs") - pl.col("gap_before_cfs")).alias("change_cfs"))
    rows = {r["canal_open"]: r for r in by_state.to_dicts()}
    open_, closed = rows.get(True, {}), rows.get(False, {})
    diff = (open_.get("median_gap_cfs") or 0) - (closed.get("median_gap_cfs") or 0)
    # The canal explains the gap only if open-season gap is clearly larger AND
    # the gap rises at openings and falls at closings.
    consistent = 0
    if ev.height:
        consistent = int(((ev["event"] == "opens") & (ev["change_cfs"] > 0)).sum()
                         + ((ev["event"] == "closes") & (ev["change_cfs"] < 0)).sum())
    return {"by_state": by_state, "events": ev, "open_minus_closed_cfs": diff,
            "events_consistent": consistent, "events_total": ev.height,
            "supported": bool(diff > 300 and ev.height and consistent / ev.height >= 0.75)}


def monthly_gap(daily: pl.DataFrame) -> pl.DataFrame:
    return (daily.group_by(pl.col("date").dt.truncate("1mo").alias("month"))
            .agg(pl.col("gap_cfs").median().alias("median_gap_cfs"),
                 pl.col("gap_pct").median().alias("median_gap_pct"), pl.len().alias("days"))
            .sort("month"))


def offset_periods(monthly: pl.DataFrame, threshold_cfs: float = 1000, min_months: int = 4,
                   max_break: int = 3) -> list[dict]:
    """Stretches of months whose median gap is above `threshold_cfs`.

    Up to `max_break` quieter months inside a stretch don't end it: a
    calibration offset shows less at low flow, so it can dip under the
    threshold in summer without having gone away.
    """
    idx = lambda d: d.year * 12 + d.month  # noqa: E731
    high = [r for r in monthly.filter(pl.col("days") >= 10).sort("month").to_dicts()
            if r["median_gap_cfs"] > threshold_cfs]
    runs, cur = [], []
    for r in high:
        if cur and idx(r["month"]) - idx(cur[-1]["month"]) > max_break + 1:
            runs.append(cur); cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return [{"start": r[0]["month"], "end": r[-1]["month"],
             "months": idx(r[-1]["month"]) - idx(r[0]["month"]) + 1,
             "median_gap_cfs": float(np.median([x["median_gap_cfs"] for x in r]))}
            for r in runs if idx(r[-1]["month"]) - idx(r[0]["month"]) + 1 >= min_months]


def calibration_check(readings: pl.DataFrame, measurements: pl.DataFrame,
                      tolerance: str = "1h") -> pl.DataFrame:
    """Each hand measurement vs. what the gauge reported closest in time.

    `measurements`: columns monitoring_location_id, time, value (cfs).
    Returns one row per measurement with the percent difference
    (gauge minus hand measurement, relative to the hand measurement).
    """
    site_map = {v["id"]: k for k, v in config.SITES.items()}
    m = (measurements.with_columns(
            pl.col("monitoring_location_id").replace_strict(site_map, default=None).alias("site"),
            pl.col("time").cast(pl.Datetime("us", "UTC")), pl.col("value").cast(pl.Float64))
         .drop_nulls(["site", "value"]).filter(pl.col("value") > 0)
         .select("site", "time", pl.col("value").alias("measured_cfs")).sort("site", "time"))
    g = (readings.filter(pl.col("param") == config.FLOW).drop_nulls("value")
         .select("site", "time", pl.col("value").alias("gauge_cfs"), "bad").sort("site", "time"))
    j = m.join_asof(g, on="time", by="site", strategy="nearest", tolerance=tolerance,
                    check_sortedness=False)
    return (j.drop_nulls("gauge_cfs")
            .with_columns((100 * (pl.col("gauge_cfs") - pl.col("measured_cfs")) / pl.col("measured_cfs"))
                          .alias("gauge_error_pct"))
            .sort("site", "time"))


def calibration_summary(cal: pl.DataFrame, periods: list[dict]) -> pl.DataFrame:
    """Median gauge error per site, inside vs. outside the suspect periods."""
    if cal.height == 0:
        return pl.DataFrame()
    inside = pl.lit(False)
    for p in periods:
        end = (p["end"].replace(day=28) + timedelta(days=4)).replace(day=1)
        inside = inside | pl.col("time").dt.date().is_between(p["start"], end - timedelta(days=1))
    return (cal.with_columns(inside.alias("in_offset_period"))
            .group_by("site", "in_offset_period")
            .agg(pl.len().alias("measurements"), pl.col("gauge_error_pct").median().alias("median_error_pct"))
            .sort("site", "in_offset_period"))


def plot_monthly(monthly: pl.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    m = monthly.filter(pl.col("days") >= 10)
    for y, (o, c) in CANAL_SEASONS.items():
        ax.axvspan(date.fromisoformat(o), date.fromisoformat(c), color="#bee3f8", alpha=0.5, lw=0,
                   label="Canal navigation season" if y == min(CANAL_SEASONS) else None)
    colors = ["#c05621" if v > 1000 else "#718096" for v in m["median_gap_cfs"]]
    ax.bar(m["month"], m["median_gap_cfs"], width=24, color=colors, align="edge")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Median daily gap (cfs)")
    ax.set_title("Green Island minus upstream gauges, by month")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(balance: pl.DataFrame, readings: pl.DataFrame, measurements: pl.DataFrame | None, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    daily = daily_gap(balance)
    canal = canal_test(daily)
    monthly = monthly_gap(daily)
    periods = offset_periods(monthly)
    plot_monthly(monthly, out / "monthly_gap.png")

    cal = calibration_check(readings, measurements) if measurements is not None and measurements.height else pl.DataFrame()
    cal_sum = calibration_summary(cal, periods)
    if cal.height:
        cal.write_csv(out / "calibration_check.csv")

    s = {r["canal_open"]: r for r in canal["by_state"].to_dicts()}
    result = {
        "canal_supported": canal["supported"],
        "gap_open_cfs": s.get(True, {}).get("median_gap_cfs"),
        "gap_closed_cfs": s.get(False, {}).get("median_gap_cfs"),
        "pct_open": s.get(True, {}).get("median_gap_pct"),
        "pct_closed": s.get(False, {}).get("median_gap_pct"),
        "flow_open_cfs": s.get(True, {}).get("median_outflow_cfs"),
        "flow_closed_cfs": s.get(False, {}).get("median_outflow_cfs"),
        "events_consistent": canal["events_consistent"], "events_total": canal["events_total"],
        "offset_periods": [{**p, "start": str(p["start"]), "end": str(p["end"])} for p in periods],
        "calibration": cal_sum.to_dicts() if cal_sum.height else [],
        "n_measurements": cal.height,
    }
    (out / "investigation.json").write_text(json.dumps(result, indent=2, default=str))
    return result
