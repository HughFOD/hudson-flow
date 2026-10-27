"""Realistic fake data with known, planted problems.

Used by the tests and by `python run.py demo`, so you can check the whole
pipeline works (and that every check catches what it should) before touching
real data. The river output has exactly the same columns as the USGS
download, and the rainfall has the same columns as `weather.fetch_rain`.

Storms are driven by rainfall: rain on each basin is routed to its gauge
through a unit hydrograph (a delayed, smeared response), so rain leads flow
by about a day. That's what makes multi-hour forecasting possible.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import polars as pl

from . import config
from .chemistry import do_saturation_mgL

TRUE_LAGS = {"upper_hudson": 3, "mohawk": 2}   # river travel time to Green Island, hours
BASIN_PEAK_STEPS = {"upper_hudson": 96, "mohawk": 64}  # rain -> gauge peak: 24 h, 16 h
UNGAUGED_FRACTION = 0.03                        # small creeks between gauges

# Known faults, so tests can check they're caught.  (site, param, start step, length)
FLATLINE = ("mohawk", "discharge_cfs", 1000, 30)
SPIKE = ("green_island", "discharge_cfs", 2000)
ICE = ("upper_hudson", "discharge_cfs", 300, 200)
GAP = ("green_island", "discharge_cfs", 2500, 40)
# A slow calibration drift with NO qualifier: the sensor-level rules can't see
# it. Only a physics check (or a model of normal behaviour) can.
DRIFT = ("mohawk", "discharge_cfs", 3000, 672, 1.4)   # ramps to +40% over 7 days


def _rain(n, rng, n_events):
    r = np.zeros(n)
    for _ in range(n_events):
        start, dur = rng.integers(0, n), int(rng.integers(16, 48))
        total = rng.gamma(2.0, 10.0)                    # mm, mostly 5-40
        r[start:start + dur] += total / dur
    return r


def _unit_hydrograph(peak_steps):
    t = np.arange(10 * peak_steps)
    k = (t / peak_steps) * np.exp(1 - t / peak_steps)
    return k / k.sum()


@lru_cache(maxsize=8)
def _simulate(days: int, seed: int):
    rng = np.random.default_rng(seed)
    n = days * 96  # 15-minute steps
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=15 * i) for i in range(n)]
    step = np.arange(n)

    common = _rain(n, rng, max(1, days // 5))
    rain = {
        "upper_hudson": common * rng.uniform(0.6, 1.4) + _rain(n, rng, max(1, days // 12)),
        "mohawk": common * rng.uniform(0.6, 1.4) + _rain(n, rng, max(1, days // 12)),
    }
    base = {"upper_hudson": 5000.0, "mohawk": 4000.0}
    q = {}
    for s in rain:
        seasonal = base[s] * (1 + 0.4 * np.sin(2 * np.pi * step / (96 * 365) + 1.0))
        runoff = np.convolve(rain[s], _unit_hydrograph(BASIN_PEAK_STEPS[s]))[:n]
        q[s] = seasonal + base[s] * 25 * runoff
    out = sum(np.roll(q[s], lag * 4) for s, lag in TRUE_LAGS.items())
    q["green_island"] = out * (1 + UNGAUGED_FRACTION) * (1 + rng.normal(0, 0.01, n))

    cond = {}
    for s in ("upper_hudson", "mohawk"):  # conductance dilutes during storms
        c0 = 150.0 if s == "upper_hudson" else 420.0
        cond[s] = c0 * (q[s].mean() / q[s]) ** 0.3
    qc = sum(np.roll(q[s] * cond[s], TRUE_LAGS[s] * 4) for s in TRUE_LAGS)
    qq = sum(np.roll(q[s], TRUE_LAGS[s] * 4) for s in TRUE_LAGS)
    cond["green_island"] = qc / qq

    temp = 4 + 2 * np.sin(2 * np.pi * step / 96) + rng.normal(0, 0.1, n)
    do = do_saturation_mgL(temp) * 0.95
    do[-500:] *= 3.0  # fouled DO sensor at the end: impossible supersaturation

    values = {}
    for s in config.SITES:
        values[(s, "discharge_cfs")] = q[s] * (1 + rng.normal(0, 0.003, n))
        values[(s, "spec_cond_uScm")] = cond[s] * (1 + rng.normal(0, 0.005, n))
    values[("green_island", "temp_C")] = temp
    values[("green_island", "do_mgL")] = do

    code = {v: k for k, v in config.PARAMS.items()}
    frames = []
    for (s, p), v in values.items():
        v = v.copy()
        qual = np.array([""] * n, dtype=object)
        n_prov = min(960, n // 4)
        approval = np.array(["Approved"] * (n - n_prov) + ["Provisional"] * n_prov, dtype=object)
        keep = np.ones(n, dtype=bool)
        if (s, p) == FLATLINE[:2]:
            v[FLATLINE[2]:FLATLINE[2] + FLATLINE[3]] = v[FLATLINE[2]]
        if (s, p) == SPIKE[:2]:
            v[SPIKE[2]] *= 4
        if (s, p) == ICE[:2]:
            qual[ICE[2]:ICE[2] + ICE[3]] = "ICE"
            v[ICE[2]:ICE[2] + ICE[3]] *= 0.6  # ice backs water up -> bad reading
        if (s, p) == GAP[:2]:
            keep[GAP[2]:GAP[2] + GAP[3]] = False
        if (s, p) == DRIFT[:2] and DRIFT[2] + DRIFT[3] <= n:
            a, L, mx = DRIFT[2], DRIFT[3], DRIFT[4]
            v[a:a + L] *= np.linspace(1.0, mx, L)
        frames.append(pl.DataFrame({
            "monitoring_location_id": config.SITES[s]["id"],
            "parameter_code": code[p],
            "time": [times[i] for i in range(n) if keep[i]],
            "value": v[keep],
            "unit_of_measure": "",
            "approval_status": approval[keep],
            "qualifier": qual[keep],
            "last_modified": "2025-06-01",
        }))
    raw = pl.concat(frames)

    # A conflicting duplicate: same time, different value, newer edit wins
    dup = raw.filter(
        (pl.col("monitoring_location_id") == config.SITES["mohawk"]["id"])
        & (pl.col("parameter_code") == "00060")
    ).slice(500, 1).with_columns(pl.col("value") * 1.1, pl.lit("2025-07-01").alias("last_modified"))
    raw = pl.concat([raw, dup]).with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))

    # Hourly rain: rain falling during [H, H+1h) is stamped H+1h, matching
    # Open-Meteo's "total over the preceding hour" convention.
    rain_h = (
        pl.DataFrame({"time": times, **{f"rain_{b}": r for b, r in rain.items()}})
        .with_columns(pl.col("time").cast(pl.Datetime("us", "UTC"))
                      .dt.truncate("1h").dt.offset_by("1h"))
        .group_by("time").agg(pl.all().sum()).sort("time")
    )
    return raw, rain_h


def make_raw(days: int = 40, seed: int = 0) -> pl.DataFrame:
    return _simulate(days, seed)[0]


def make_rain(days: int = 40, seed: int = 0) -> pl.DataFrame:
    return _simulate(days, seed)[1]
