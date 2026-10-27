"""Command-line entry point.

    python run.py demo                     # everything, on synthetic data (no internet), ~3 min
    python run.py demo --no-ml             # just the validation pipeline, seconds
    python run.py inventory                # what does each gauge measure?
    python run.py fetch   --start 2022-01-01 --end 2025-12-31
    python run.py analyze --start 2022-01-01 --end 2025-12-31
    python run.py investigate --start 2022-01-01 --end 2025-12-31   # why the balance doesn't close

Use at least 3 years of real data for the forecasting: the models need to
see every season more than once.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from hudson import (anomaly, balance, chemistry, clean, fetch, forecast, investigate, ml_report,
                    report, synthetic, weather)

CACHE = Path("data/raw")


def analyze(raw: pl.DataFrame, out: Path, rain: pl.DataFrame | None, ml: bool) -> None:
    print(f"Cleaning {raw.height:,} raw readings...")
    cleaned = clean.clean(raw)
    print(cleaned["summary"])
    hourly = clean.to_hourly(cleaned["readings"])

    bal, bal_summary = balance.flow_balance(hourly)
    print("Flow balance:", bal_summary)
    monthly = balance.monthly_residuals(bal)
    cb = chemistry.conductance_balance(hourly, bal_summary["lags"])
    do = chemistry.do_check(hourly)
    path = report.write_report(out, cleaned, hourly, bal, bal_summary, monthly, cb, do)
    print(f"\nValidation report: {path}")
    if not ml:
        return

    print("\nForecasting...")
    hourly_r = weather.attach_rain(hourly, rain)
    results = forecast.run(hourly_r, bal_summary["lags"])
    for r in results:
        print(r["table"].select("horizon_h", "model", "MAE_cfs", "NSE", "skill_vs_persistence"))

    print("Anomaly detection...")
    hourly_all = clean.to_hourly(cleaned["readings"], include_bad=True)
    det = anomaly.detect(anomaly.build_signals(hourly_all, bal_summary["lags"]))
    compared, agree = anomaly.compare(det, anomaly.rule_flags_hourly(cleaned["readings"], bal))
    eps = anomaly.episodes(compared)
    print(agree)

    path = ml_report.write(out, results, compared, agree, eps, bal, bal_summary,
                           has_rain=rain is not None)
    print(f"ML report: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["demo", "inventory", "fetch", "analyze", "investigate"])
    p.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--no-ml", action="store_true", help="skip forecasting and anomaly detection")
    a = p.parse_args()

    if a.command == "demo":
        days = 730
        analyze(synthetic.make_raw(days), a.out / "demo", synthetic.make_rain(days), ml=not a.no_ml)
    elif a.command == "inventory":
        with pl.Config(tbl_rows=200, fmt_str_lengths=60):
            print(fetch.inventory())
    elif a.command == "fetch":
        fetch.fetch_all(a.start, a.end, CACHE)
        weather.fetch_rain(a.start, a.end, CACHE)
    elif a.command == "analyze":
        raw = fetch.fetch_all(a.start, a.end, CACHE)  # reads from cache when it can
        raw = raw.filter(pl.col("time").dt.date().is_between(a.start, a.end))
        rain = None
        if not a.no_ml:
            try:
                rain = weather.fetch_rain(a.start, a.end, CACHE)
            except Exception as exc:
                print(f"! Rainfall unavailable ({type(exc).__name__}: {exc}); forecasting without it")
        analyze(raw, a.out / f"{a.start}_{a.end}", rain, ml=not a.no_ml)
    elif a.command == "investigate":
        run_dir = a.out / f"{a.start}_{a.end}"
        bal_path, rd_path = run_dir / "flow_balance.parquet", run_dir / "readings_flagged.parquet"
        if not bal_path.exists():
            raise SystemExit("Run the analysis first (step3_run_real_data.py).")
        try:
            meas = fetch.fetch_measurements(a.start, a.end, CACHE)
        except Exception as exc:
            print(f"! Field measurements unavailable ({type(exc).__name__}: {exc})")
            meas = None
        res = investigate.run(pl.read_parquet(bal_path), pl.read_parquet(rd_path), meas, run_dir)
        print(f"Canal explains the gap: {res['canal_supported']}  "
              f"(gap {res['gap_open_cfs']:.0f} cfs canal open vs {res['gap_closed_cfs']:.0f} cfs closed)")
        for p in res["offset_periods"]:
            print(f"Offset period: {p['start']} to {p['end']}, median gap {p['median_gap_cfs']:.0f} cfs")
        for r in res["calibration"]:
            print(f"  {r['site']:<13} in offset period={r['in_offset_period']!s:<5} "
                  f"{r['measurements']:>3} hand measurements, gauge error {r['median_error_pct']:+.1f}%")
        print(f"Saved to {run_dir}/investigation.json and monthly_gap.png")


if __name__ == "__main__":
    main()
