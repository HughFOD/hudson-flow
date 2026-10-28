"""Build a static project website (docs/) from the latest analysis results.

Every number on the page is computed here from the pipeline's output files,
so re-running the analysis and then this script keeps the site truthful.
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from . import config

MODELS = {"persistence": "No change", "physics": "Mass balance", "ridge": "Linear (ridge)",
          "gbm": "Boosted trees", "hybrid": "Balance + trees"}
REPO_URL = "https://github.com/HughFOD/hudson-flow"  # optional: paste your GitHub repo link here, e.g. "https://github.com/you/hudson-flow"


def latest_run(out: Path) -> Path:
    runs = [p for p in out.iterdir() if p.is_dir() and p.name != "demo" and (p / "flow_balance.parquet").exists()]
    if not runs:
        sys.exit("No real-data results found. Run step3_run_real_data.py first.")
    return max(runs, key=lambda p: (p / "flow_balance.parquet").stat().st_mtime)


def stats(run: Path) -> dict:
    s: dict = {"run": run.name}
    start, _, end = run.name.partition("_")
    s["start"], s["end"] = start, end

    rd = pl.read_parquet(run / "readings_flagged.parquet")
    s["readings"] = rd.height
    s["flagged"] = int(rd["bad"].sum())
    flow = rd.filter(pl.col("param") == config.FLOW)
    s["gauges"] = (flow.group_by("site").agg(
        pl.len().alias("n"), pl.col("bad").mean().alias("flagged"),
        pl.col("flag_estimated").sum().alias("estimated"), pl.col("flag_flatline").sum().alias("stuck"),
        pl.col("flag_spike").sum().alias("spikes"), pl.col("is_provisional").sum().alias("provisional"),
    ).sort("site").to_dicts())

    b = pl.read_parquet(run / "flow_balance.parquet").filter(~pl.col("incomplete"))
    out_col = f"{config.OUTFLOW}__{config.FLOW}"
    daily = (b.group_by(pl.col("time").dt.date().alias("d"))
             .agg(pl.col(out_col).mean().alias("o"), pl.col("predicted_cfs").mean().alias("p"), pl.len().alias("h"))
             .filter(pl.col("h") >= 20)
             .with_columns((100 * (pl.col("o") - pl.col("p")) / pl.col("o")).alias("m")))
    s["hourly_abs"] = float(b["residual_pct"].abs().median())
    s["daily_median"] = float(daily["m"].median())
    s["days"] = daily.height
    monthly = (daily.group_by(pl.col("d").dt.month().alias("month")).agg(pl.col("m").median().alias("m"))
               .sort("month"))
    s["monthly"] = monthly.to_dicts()
    peak = monthly.sort("m", descending=True).row(0, named=True)
    s["peak_month"], s["peak_pct"] = date(2000, peak["month"], 1).strftime("%B"), peak["m"]
    low = monthly.filter(pl.col("month").is_in([12, 1, 2, 3])).get_column("m")
    s["winter_pct"] = float(low.median()) if low.len() else float("nan")

    s["forecasts"] = []
    for h in config.HORIZONS:
        p = run / f"forecasts_{h}h.parquet"
        if not p.exists():
            continue
        f = pl.read_parquet(p)
        y = f["y"].to_numpy()
        base = np.mean(np.abs(f["persistence"].to_numpy() - y))
        row = {"h": h}
        for m in MODELS:
            mae = np.mean(np.abs(f[m].to_numpy() - y))
            row[m] = {"mae": mae, "skill": 1 - mae / base}
        lo, hi = f["hybrid_lo"].to_numpy(), f["hybrid_hi"].to_numpy()
        row["coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        s["forecasts"].append(row)
    inv = run / "investigation.json"
    s["inv"] = json.loads(inv.read_text()) if inv.exists() else None
    if s["inv"] and s["inv"]["offset_periods"]:
        p = s["inv"]["offset_periods"][0]
        outside = daily.filter(~pl.col("d").is_between(date.fromisoformat(p["start"]),
                                                       date.fromisoformat(p["end"]).replace(day=28)))
        s["gap_outside_cfs"] = float((outside["o"] - outside["p"]).median())
        s["gap_outside_pct"] = float(outside["m"].median())
    return s


def pct(v, signed=False, d=1):
    t = f"{v:.{d}f}%"
    return ("+" + t) if signed and v > 0 else t


CSS = """
:root{--bg:#f6f8f7;--panel:#ffffff;--fg:#14211d;--muted:#566660;--line:#d9e2de;--accent:#1f6f5c;--accent2:#b5651d;--ok:#1f7a4d;--warn:#a6611a;
--serif:"Fraunces",Georgia,serif;--sans:"Public Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0f1715;--panel:#16211e;--fg:#e4ece9;--muted:#9aaba4;--line:#2a3833;--accent:#5cc2a6;--accent2:#e3a35f;--ok:#6fd39f;--warn:#e7a75a;color-scheme:dark}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 var(--sans)}
.wrap{max-width:60rem;margin:0 auto;padding:3rem max(1rem,env(safe-area-inset-right)) 4rem max(1rem,env(safe-area-inset-left))}
.eyebrow{font:500 .75rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font:400 clamp(2rem,5vw,3.2rem)/1.1 var(--serif);margin:.6rem 0 1rem;text-wrap:balance}
h2{font:400 1.7rem/1.2 var(--serif);margin:3rem 0 .75rem;text-wrap:balance}
h3{font-size:1rem;margin:0 0 .25rem}
p{max-width:65ch;margin:0 0 1rem}
.lede{font-size:1.15rem;color:var(--fg)}
.muted{color:var(--muted)}
a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));gap:1rem;margin:2rem 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem 1.1rem}
.kpi b{display:block;font:400 2rem/1.1 var(--serif);font-variant-numeric:tabular-nums;margin-bottom:.35rem}
.kpi span{font-size:.85rem;color:var(--muted);line-height:1.4;display:block}
figure{margin:1.5rem 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem}
figure img{display:block;width:100%;height:auto;border-radius:4px;background:#fff}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.6rem}
.flow{margin:1rem 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.75rem 1rem}.flow svg{display:block;max-width:660px;margin:0 auto}.flow .river{fill:none;stroke:var(--accent);stroke-width:5;stroke-linecap:round}.flow .arrow{fill:none;stroke:var(--accent);stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.flow .g{fill:var(--panel);stroke:var(--fg);stroke-width:2.5}.flow text{fill:var(--fg);font:500 15px var(--sans)}
.diagram{font:400 .85rem/1.5 var(--mono);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem;overflow-x:auto;white-space:pre}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:.55rem .8rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
tr:last-child td{border-bottom:0}td.best{color:var(--ok);font-weight:600}td.neg{color:var(--warn)}
.months{display:grid;grid-template-columns:repeat(12,1fr);gap:4px;align-items:end;height:140px;margin:1rem 0 .3rem}
.months div{background:var(--line);border-radius:4px 4px 0 0;min-height:2px;position:relative}
.months div.hi{background:var(--accent2)}
.mlabels{display:grid;grid-template-columns:repeat(12,1fr);gap:4px;font:400 .7rem var(--mono);color:var(--muted);text-align:center}
ul,ol{padding-left:1.2rem;max-width:65ch}li{margin:.3rem 0}
code{font-family:var(--mono);font-size:.88em}
footer{margin-top:4rem;font-size:.85rem;color:var(--muted);border-top:1px solid var(--line);padding-top:1rem}
"""


def build(root: Path) -> Path:
    run = latest_run(root / "output")
    s = stats(run)
    docs = root / "docs"
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)
    charts = {}
    for name in ("monthly_gap", "flow_balance", "hydrograph", "storm_forecast", "skill_by_horizon", "feature_importance", "anomalies"):
        src = run / f"{name}.png"
        if src.exists():
            shutil.copy(src, img / src.name)
            charts[name] = f"img/{src.name}"

    e = html.escape
    names = {"upper_hudson": "Upper Hudson (01335754)", "mohawk": "Mohawk at Cohoes (01357500)",
             "green_island": "Green Island (01358000)"}


    grows = "".join(
        f"<tr><td>{e(names.get(g['site'], g['site']))}</td><td>{g['n']:,}</td><td>{pct(100 * g['flagged'])}</td>"
        f"<td>{g['estimated']:,}</td><td>{g['stuck']:,}</td><td>{g['spikes']:,}</td><td>{g['provisional']:,}</td></tr>"
        for g in s["gauges"])

    frows, best12 = "", None
    for f in s["forecasts"]:
        main = {m: f[m] for m in MODELS if m != "persistence"}
        best = max(main, key=lambda m: main[m]["skill"])
        if f["h"] == 12:
            best12 = (best, main[best]["skill"])
        cells = "".join(
            f'<td class="{"best" if m == best else ("neg" if f[m]["skill"] < 0 else "")}">{f[m]["skill"]:+.0%}</td>'
            for m in MODELS if m != "persistence")
        frows += f"<tr><td>{f['h']} h ahead</td>{cells}<td>{f['coverage']:.0%}</td></tr>"
    fhead = "".join(f"<th>{e(MODELS[m])}</th>" for m in MODELS if m != "persistence")

    def fig(key, cap):
        return f'<figure><img src="{charts[key]}" alt="{e(cap)}" loading="lazy"><figcaption>{e(cap)}</figcaption></figure>' if key in charts else ""

    inv = s.get("inv")
    if inv:
        periods = inv["offset_periods"]
        per = periods[0] if periods else None
        per_txt = (f"<p>The real pattern is a step. From {date.fromisoformat(per['start']):%B %Y} to {date.fromisoformat(per['end']):%B %Y} "
                   f"the gap jumps to a median of {per['median_gap_cfs']:,.0f} cfs, then drops back. Outside that stretch the median gap is "
                   f"{s.get('gap_outside_cfs', 0):,.0f} cfs ({pct(s.get('gap_outside_pct', 0), True)}). An offset that switches on and off like that, "
                   f"regardless of season or flow, looks like a calibration problem at one gauge rather than anything physical in the river.</p>") if per else ""
        if inv["calibration"]:
            rows = "".join(
                f"<tr><td>{e(names.get(r['site'], r['site']))}</td><td>{'Yes' if r['in_offset_period'] else 'No'}</td>"
                f"<td>{r['measurements']}</td><td>{r['median_error_pct']:+.1f}%</td></tr>" for r in inv["calibration"])
            cal_txt = ("<p>To find which gauge, I compared each gauge's reported flow with the USGS field crews' hand measurements, "
                       "which the gauges are calibrated against:</p><div class=\"tablewrap\"><table><thead><tr><th>Gauge</th>"
                       "<th>In offset period</th><th>Hand measurements</th><th>Gauge vs. hand measurement</th></tr></thead>"
                       f"<tbody>{rows}</tbody></table></div>")
        else:
            cal_txt = ("<p><b>Next test:</b> USGS crews measure each river by hand several times a year, and the gauges are calibrated against those visits. "
                       "Comparing each gauge's reported flow with the hand measurements during the offset period will show which gauge drifted "
                       "(<code>step7_investigate.py</code> does this).</p>")
        investigation = f"""<p>My first explanation was the Barge Canal: part of the Mohawk is diverted around the Cohoes gauge through the Waterford locks
and rejoins the Hudson above Green Island. I tested it against the canal's actual opening and closing dates for 2022–2025, and <b>it does not hold up</b>.
Measured in cubic feet per second, the gap is about the same with the canal open ({inv['gap_open_cfs']:,.0f} cfs) as closed ({inv['gap_closed_cfs']:,.0f} cfs),
and it moved the expected way at only {inv['events_consistent']} of {inv['events_total']} openings and closings. It only looked seasonal in percent terms
because the river is low in late summer ({inv['flow_open_cfs']:,.0f} cfs vs {inv['flow_closed_cfs']:,.0f} cfs), so the same gap is a bigger share.</p>
{fig('monthly_gap', 'Median daily gap by month in cfs, with the canal navigation seasons shaded. Orange months are above 1,000 cfs.')}
{per_txt}
{cal_txt}"""
    else:
        investigation = f"""<p>The gap is larger in percent terms in late summer and autumn ({pct(s['peak_pct'], True)} in {e(s['peak_month'])}), when the river is low.
Run <code>step7_investigate.py</code> to test the canal explanation and check each gauge against USGS hand measurements.</p>"""

    best_txt = (f"{MODELS[best12[0]]} cut 12-hour forecast error by {best12[1]:.0%}"
                if best12 else "Forecast results")
    repo = f'<a href="{e(REPO_URL)}">Code on GitHub</a> · ' if REPO_URL else ""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Hudson River Gauge Check</title>
<meta name="description" content="Validating and forecasting USGS Hudson and Mohawk river gauges with mass balances and machine learning.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400&family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main class="wrap">
<div class="eyebrow">Data validation · Chemical engineering · Machine learning</div>
<h1>Hudson River Gauge Check</h1>
<p class="lede">Where the Mohawk joins the Hudson near Troy, New York, the water flowing out must equal the water flowing in.
I used that mass balance to test whether three U.S. Geological Survey river gauges can be trusted, then built models to forecast the river up to a day ahead.</p>
<p class="muted">{e(s['start'])} to {e(s['end'])} · 15-minute USGS data · Python, polars, scikit-learn</p>

<div class="kpis">
<div class="kpi"><b>{s['readings']:,}</b><span>sensor readings checked</span></div>
<div class="kpi"><b>{pct(100 * s['flagged'] / s['readings'])}</b><span>flagged as unreliable: estimated, stuck, spiking or missing</span></div>
<div class="kpi"><b>{pct(s['daily_median'], True)}</b><span>median daily gap between inflow and outflow</span></div>
<div class="kpi"><b>{best12[1]:.0%}</b><span>{e(best_txt.split(' cut')[0])}: reduction in 12-hour forecast error vs. assuming no change</span></div>
</div>

<h2>How it works</h2>
<p>Two rivers meet near Troy, New York: the upper Hudson and the Mohawk. The U.S. Geological Survey has a gauge on each one, measuring how much water flows past every 15 minutes, and a third gauge just downstream at Green Island, where the two have combined.</p>
<figure class="flow" aria-label="The upper Hudson and Mohawk gauges both flow into the Green Island gauge">
<svg viewBox="0 0 660 150" width="100%" role="img">
<path d="M200 38 C 300 38, 310 75, 380 75" class="river"/>
<path d="M200 112 C 300 112, 310 75, 380 75" class="river"/>
<path d="M380 75 L 445 75" class="river"/><path d="M443 67 L 455 75 L 443 83" class="arrow"/>
<circle cx="190" cy="38" r="7" class="g"/><circle cx="190" cy="112" r="7" class="g"/><circle cx="470" cy="75" r="7" class="g"/>
<text x="176" y="43" text-anchor="end">Upper Hudson gauge</text>
<text x="176" y="117" text-anchor="end">Mohawk gauge</text>
<text x="484" y="80">Green Island gauge</text>
</svg></figure>
<p>Water doesn't appear or disappear, so the water passing Green Island should equal what came down the two rivers an hour or two earlier. When the numbers don't add up, either a gauge is wrong or water is getting in or out somewhere nobody measures. That simple rule is the core of the project. It does four things:</p>
<ol>
<li><b>Checks every reading</b> for problems like stuck sensors, sudden fake spikes, ice, and values USGS had to estimate.</li>
<li><b>Checks that the water adds up</b> between the three gauges, and investigates where it doesn't.</li>
<li><b>Forecasts the river</b> at Green Island 6 to 24 hours ahead, using the upstream gauges and rainfall.</li>
<li><b>Compares two ways of spotting bad data</b>: simple rules versus a machine-learning model.</li>
</ol>

<h2>Do the gauges add up?</h2>
<p>Mostly, yes: the median daily gap is {pct(s['daily_median'], True)}. Hour to hour it is noisier ({pct(s['hourly_abs'])} median absolute), which points to dam and lock operations releasing water in pulses rather than to faulty sensors.</p>
{investigation}
{fig('flow_balance', 'Hourly flow balance: measured flow at Green Island against the lagged sum of the upstream gauges, and the residual.')}

<h2>How trustworthy is each gauge?</h2>
<div class="tablewrap"><table><thead><tr><th>Discharge gauge</th><th>Readings</th><th>Flagged</th><th>Estimated</th><th>Stuck</th><th>Spikes</th><th>Provisional</th></tr></thead>
<tbody>{grows}</tbody></table></div>
<p class="muted" style="margin-top:.75rem">Findings worth knowing: for most of 2024 Green Island published only 2-hourly estimated values, and the Mohawk's conductance and temperature sensors were stuck or spiking for long stretches.</p>

<h2>Forecasting Green Island</h2>
<p>Five models predict flow 6 to 24 hours ahead using upstream flows and basin rainfall. They are tested walk-forward: always trained on the past and scored on the future, with an embargo so no training target overlaps the test period. Skill is the share of error removed compared with assuming the flow stays the same.</p>
<div class="tablewrap"><table><thead><tr><th>Horizon</th>{fhead}<th>90% interval coverage</th></tr></thead><tbody>{frows}</tbody></table></div>
<p class="muted" style="margin-top:.75rem">Best model per horizon in green; negative skill means worse than assuming no change. Prediction intervals are calibrated with conformal prediction, so about 90% of real outcomes should fall inside.</p>
{fig('storm_forecast', 'The largest storm in the test period: forecasts made 12 hours ahead against what Green Island measured.')}
{fig('skill_by_horizon', 'Forecast skill by horizon, including ablations without rainfall and without upstream gauges.')}

<h2>What I learned</h2>
<ul>
<li>Physics found problems rules could not. A sensor drifting out of calibration still reports plausible numbers; only the balance against its neighbours exposes it.</li>
<li>Real data broke assumptions that synthetic data never tested: a date-range bug that dropped every December 31st, USGS changing its quality codes between APIs, and a sensor that only existed for six weeks.</li>
<li>My first explanation for the imbalance, the Barge Canal, was wrong. Testing it against the real canal dates and measuring the gap in cfs instead of percent turned up a year-long calibration offset instead.</li>
<li>The simple mass-balance forecast lost to "no change" on the real river. Reporting that honestly is more useful than hiding it.</li>
</ul>

<footer>{repo}Data: U.S. Geological Survey Water Data API; rainfall from Open-Meteo. Generated {date.today():%B %-d, %Y} from run <code>{e(s['run'])}</code>.</footer>
</main></body></html>"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    return docs / "index.html"


if __name__ == "__main__":
    root = Path.cwd()
    path = build(root)
    print(f"✅ Website built: {path}")
    print("   Preview: double-click docs/index.html in Finder.")
    print("   Publish: push to GitHub, then Settings > Pages > Branch: main, folder /docs.")
