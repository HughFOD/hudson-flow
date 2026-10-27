"""Step 5: turn the results into spreadsheet files (.csv) you can open in
Excel, Numbers or Google Sheets.

In PyCharm: click the green ▶ arrow next to `if __name__ == "__main__":` below.
The files appear in output/<run folder>/spreadsheets/. Double-click one in
Finder to open it in Excel or Numbers.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

CODE = r'''
import polars as pl
from pathlib import Path

for run in sorted(Path("output").iterdir()):
    if not run.is_dir():
        continue
    out = run / "spreadsheets"
    out.mkdir(exist_ok=True)
    print(f"\n{run.name}:")

    p = run / "flow_balance.parquet"
    if p.exists():
        df = pl.read_parquet(p).with_columns(pl.col("time").dt.strftime("%Y-%m-%d %H:%M"))
        df.write_csv(out / "hourly_flow_balance.csv")
        print(f"  hourly_flow_balance.csv  ({df.height:,} rows: every hour, all 3 gauges, the mismatch)")
        daily = (pl.read_parquet(p).filter(~pl.col("incomplete"))
                 .group_by(pl.col("time").dt.date().alias("date"))
                 .agg(pl.col("^.*_cfs$").mean().round(0), pl.len().alias("hours"))
                 .filter(pl.col("hours") >= 20)
                 .with_columns((100 * pl.col("residual_cfs") / pl.col("green_island__discharge_cfs"))
                               .round(1).alias("daily_mismatch_pct"))
                 .sort("date"))
        daily.write_csv(out / "daily_flow_balance.csv")
        print(f"  daily_flow_balance.csv   ({daily.height:,} rows: one row per day)")

    p = run / "readings_flagged.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        bad = df.filter(pl.col("bad")).with_columns(pl.col("time").dt.strftime("%Y-%m-%d %H:%M"))
        bad.drop("monitoring_location_id", "parameter_code").write_csv(out / "flagged_readings.csv")
        print(f"  flagged_readings.csv     ({bad.height:,} rows: every reading the checks flagged, and why)")

    for p in sorted(run.glob("forecasts_*h.parquet")):
        df = pl.read_parquet(p).with_columns(pl.col("time").dt.strftime("%Y-%m-%d %H:%M"))
        name = p.stem + ".csv"
        df.write_csv(out / name)
        print(f"  {name:<24} ({df.height:,} rows: each model's forecast vs what happened)")

print("\nDone. In Finder: PycharmProjects > Hudson > output > (run) > spreadsheets")
'''

if __name__ == "__main__":
    if not PY.exists():
        sys.exit("Run step1_setup.py first.")
    sys.exit(subprocess.call([str(PY), "-c", CODE], cwd=HERE))
