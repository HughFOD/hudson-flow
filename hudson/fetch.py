"""Download USGS continuous (15-minute) data and cache it as parquet.

Uses the modernized USGS Water Data API through `dataretrieval.waterdata`.
The legacy NWIS services (`dataretrieval.nwis`) are being decommissioned,
so don't use them.

The API allows at most ~3 years per call, so we pull one calendar year at a
time and cache each piece. Re-running only downloads what's missing.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from . import config

RAW_COLUMNS = [
    "monitoring_location_id", "parameter_code", "time", "value",
    "unit_of_measure", "approval_status", "qualifier", "last_modified",
]


def _year_windows(start: date, end: date):
    """One query window per calendar year.

    The API reads a bare end date as MIDNIGHT at the start of that day, so a
    window ending "2022-12-31" silently drops almost all of December 31st.
    (The first version of this code had exactly that bug: every year was
    missing its last day.) So each window runs to midnight of the NEXT day;
    the one overlapping reading is removed later by deduplication.
    """
    y = start.year
    while y <= end.year:
        s = max(start, date(y, 1, 1))
        e = min(end, date(y, 12, 31)) + timedelta(days=1)
        yield y, f"{s.isoformat()}T00:00:00Z/{e.isoformat()}T00:00:00Z"
        y += 1


def _to_polars(df) -> pl.DataFrame:
    """Convert the pandas frame from dataretrieval into a typed polars frame."""
    df = df.copy()
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[RAW_COLUMNS]
    # qualifier can come back as a list, None, or a string: normalize to text
    df["qualifier"] = df["qualifier"].apply(
        lambda q: ",".join(map(str, q)) if isinstance(q, (list, tuple)) else (q or "")
    ).astype(str)
    for c in ("monitoring_location_id", "parameter_code", "unit_of_measure",
              "approval_status"):
        df[c] = df[c].astype(str)
    df["last_modified"] = df["last_modified"].astype(str)
    out = pl.from_pandas(df)
    return out.with_columns(
        pl.col("time").cast(pl.Datetime("us", "UTC"), strict=False),
        pl.col("value").cast(pl.Float64, strict=False),
    )


def fetch_one(site_id: str, pcode: str, start: date, end: date,
              cache_dir: Path) -> pl.DataFrame:
    from dataretrieval import waterdata  # imported lazily: tests don't need it

    cache_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for year, window in _year_windows(start, end):
        # "v2": cache files from before the year-end fix are ignored and refetched
        path = cache_dir / f"{site_id}_{pcode}_{year}_v2.parquet"
        # Always refresh the current year: provisional data keeps changing.
        if path.exists() and year < date.today().year:
            pieces.append(pl.read_parquet(path))
            continue
        try:
            df, _ = waterdata.get_continuous(
                monitoring_location_id=site_id,
                parameter_code=pcode,
                time=window,
            )
        except Exception as exc:  # network errors, service outages
            print(f"  ! {site_id} {pcode} {year}: {type(exc).__name__}: {exc}")
            continue
        if df is None or len(df) == 0:
            print(f"  - {site_id} {pcode} {year}: no data")
            continue
        piece = _to_polars(df)
        piece.write_parquet(path)
        print(f"  + {site_id} {pcode} {year}: {piece.height:,} rows")
        pieces.append(piece)
    if not pieces:
        return pl.DataFrame()
    return pl.concat(pieces, how="vertical_relaxed")


def fetch_all(start: date, end: date, cache_dir: Path,
              params: dict | None = None) -> pl.DataFrame:
    """Pull every configured site x parameter. Missing combos are skipped."""
    params = params or config.PARAMS
    frames = []
    for key, site in config.SITES.items():
        for pcode in params:
            print(f"{key} ({site['id']}) {pcode} [{config.PARAMS[pcode]}]")
            df = fetch_one(site["id"], pcode, start, end, cache_dir)
            if df.height:
                frames.append(df)
    if not frames:
        raise RuntimeError("No data downloaded. Check your connection and site IDs.")
    return pl.concat(frames, how="vertical_relaxed")


def inventory() -> pl.DataFrame:
    """Which parameters does each site actually record continuously?"""
    from dataretrieval import waterdata

    ids = [s["id"] for s in config.SITES.values()]
    df, _ = waterdata.get_time_series_metadata(monitoring_location_id=ids)
    keep = [c for c in ("monitoring_location_id", "parameter_code",
                        "parameter_name", "computation_period_identifier",
                        "begin", "end") if c in df.columns]
    df = df[keep].astype(str)
    return pl.from_pandas(df).sort(keep[:2])
