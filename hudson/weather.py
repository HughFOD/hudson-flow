"""Hourly basin-average rainfall from the Open-Meteo historical archive.

Free, no API key. Rain falls on the upper Hudson and Mohawk basins hours to
days before it reaches the gauges, which is what gives a forecast lead time
beyond the river's own travel time.

Each hourly value is the precipitation total for the PRECEDING hour, so the
value stamped 10:00 is rain from 09:00-10:00 and is known at 10:00. That
matters for leakage: features may only use values stamped <= the forecast time.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from . import config

URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_point(lat, lon, start: date, end: date) -> pl.DataFrame:
    import requests

    r = requests.get(URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "precipitation", "timezone": "UTC",
    }, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pl.DataFrame({"time": h["time"], "rain_mm": h["precipitation"]}).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC"),
        pl.col("rain_mm").cast(pl.Float64),
    )


def fetch_rain(start: date, end: date, cache_dir: Path) -> pl.DataFrame:
    """Wide frame: time, rain_<basin> (mm/hour, basin average)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"rain_{start}_{end}.parquet"
    if path.exists():
        return pl.read_parquet(path)

    out = None
    for basin, points in config.RAIN_POINTS.items():
        frames = [_fetch_point(lat, lon, start, end) for lat, lon in points]
        avg = (pl.concat(frames).group_by("time").agg(pl.col("rain_mm").mean())
               .rename({"rain_mm": f"rain_{basin}"}))
        out = avg if out is None else out.join(avg, on="time", how="full", coalesce=True)
    out = out.sort("time")
    out.write_parquet(path)
    return out


def attach_rain(hourly: pl.DataFrame, rain: pl.DataFrame | None) -> pl.DataFrame:
    if rain is None:
        return hourly
    return hourly.join(rain, on="time", how="left")
