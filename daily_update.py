"""Refresh the website with the latest data. Run by GitHub Actions every day at noon New York time.

Downloads USGS readings and rainfall from January 2022 up to a week ago (the
rainfall archive runs a few days behind, and USGS revises its newest readings
for several days), reruns the analysis and the investigation, and rebuilds
docs/index.html. The design of the page does not change, only the numbers.
"""
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

START = date(2022, 1, 1)
LAG_DAYS = 7

if __name__ == "__main__":
    end = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=LAG_DAYS)
    args = ["--start", START.isoformat(), "--end", end.isoformat()]
    print(f"Updating with data from {START} to {end}", flush=True)
    for cmd in (["analyze", *args], ["investigate", *args]):
        subprocess.check_call([sys.executable, "run.py", *cmd])
    from hudson import website
    print("Website:", website.build(Path(".")))
