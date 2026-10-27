"""The investigation must find a planted gauge calibration error."""
import datetime as dt

import numpy as np
import polars as pl

from hudson import balance, clean, config, investigate, synthetic


def test_calibration_check_finds_biased_gauge(tmp_path):
    c = clean.clean(synthetic.make_raw(200))
    bal, _ = balance.flow_balance(clean.to_hourly(c["readings"]))
    flow = c["readings"].filter((pl.col("param") == config.FLOW) & ~pl.col("bad"))
    rng, rows = np.random.default_rng(0), []
    for s, v in config.SITES.items():
        f = flow.filter(pl.col("site") == s).sample(20, seed=1)
        for t, val in zip(f["time"], f["value"]):
            bias = 1.2 if s == "mohawk" else 1.0          # Mohawk gauge reads 20% high
            rows.append({"monitoring_location_id": v["id"], "time": t, "value": val / bias * rng.normal(1, 0.01)})
    res = investigate.run(bal, c["readings"], pl.DataFrame(rows), tmp_path)
    err = {r["site"]: r["median_error_pct"] for r in res["calibration"]}
    assert 15 < err["mohawk"] < 25
    assert abs(err["green_island"]) < 3 and abs(err["upper_hudson"]) < 3
    assert (tmp_path / "monthly_gap.png").exists()


def test_offset_periods_bridge_short_dips():
    months = [dt.date(2023, m, 1) for m in range(1, 13)]
    gaps = [3000, 3000, 3000, 600, 3000, 3000, 100, 100, 100, 100, 100, 100]
    m = pl.DataFrame({"month": months, "median_gap_cfs": gaps, "days": [30] * 12})
    p = investigate.offset_periods(m)
    assert len(p) == 1 and p[0]["start"] == dt.date(2023, 1, 1) and p[0]["months"] == 6
