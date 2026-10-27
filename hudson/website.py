"""Build a static project website (docs/) from the latest analysis results.

Every number on the page is computed here from the pipeline's output files,
so re-running the analysis and then this script keeps the site truthful.
"""
from __future__ import annotations

import html
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
ul{padding-left:1.2rem;max-width:65ch}li{margin:.3rem 0}
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
    for name in ("flow_balance", "hydrograph", "storm_forecast", "skill_by_horizon", "feature_importance", "anomalies"):
        src = run / f"{name}.png"
        if src.exists():
            shutil.copy(src, img / src.name)
            charts[name] = f"img/{src.name}"

    e = html.escape
    names = {"upper_hudson": "Upper Hudson (01335754)", "mohawk": "Mohawk at Cohoes (01357500)",
             "green_island": "Green Island (01358000)"}

    mx = max(abs(m["m"]) for m in s["monthly"]) or 1
    peak_m = max(s["monthly"], key=lambda m: m["m"])["month"]
    bars = "".join(
        f'<div class="{"hi" if m["month"] == peak_m else ""}" style="height:{max(2, 100 * max(0, m["m"]) / mx):.0f}%" '
        f'title="{date(2000, m["month"], 1):%B}: {pct(m["m"], True)}"></div>' for m in s["monthly"])
    labels = "".join(f"<span>{date(2000, m['month'], 1):%b}</span>" for m in s["monthly"])

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
<div class="kpi"><b>{s['readings'] / 1e6:.2f}M</b><span>sensor readings checked</span></div>
<div class="kpi"><b>{pct(100 * s['flagged'] / s['readings'])}</b><span>flagged as unreliable: estimated, stuck, spiking or missing</span></div>
<div class="kpi"><b>{pct(s['daily_median'], True)}</b><span>median daily gap between inflow and outflow</span></div>
<div class="kpi"><b>{best12[1]:.0%}</b><span>{e(best_txt.split(' cut')[0])}: reduction in 12-hour forecast error vs. assuming no change</span></div>
</div>

<h2>The setup</h2>
<div class="diagram">Upper Hudson      (USGS 01335754) ─┐
                                   ├─►  Green Island (USGS 01358000)
Mohawk at Cohoes  (USGS 01357500) ─┘

Water:  Q_GreenIsland(t) = Q_Hudson(t - lag1) + Q_Mohawk(t - lag2)
Salt:   C_GreenIsland    = (Q_Hudson*C_Hudson + Q_Mohawk*C_Mohawk) / (Q_Hudson + Q_Mohawk)</div>
<p>Each 15-minute reading is checked for USGS quality codes, stuck values, one-point spikes and gaps. Readings that pass are averaged hourly and run through the balance. Travel times between gauges are fitted jointly, since fitting each tributary on its own gives the wrong answer when one storm hits both basins.</p>

<h2>Do the gauges add up?</h2>
<p>On a typical day, yes: the median gap is {pct(s['daily_median'], True)}. Hour to hour it is noisier ({pct(s['hourly_abs'])} median absolute), which points to dam and lock operations releasing water in pulses rather than to faulty sensors.</p>
<p>The gap has a clear season. It is near {pct(s['winter_pct'], True)} in winter and climbs to {pct(s['peak_pct'], True)} in {e(s['peak_month'])}:</p>
<div class="months" aria-label="Median daily mismatch by month">{bars}</div>
<div class="mlabels">{labels}</div>
<p class="muted" style="margin-top:.75rem">Median daily mismatch by calendar month. Positive means Green Island carried more water than the two upstream gauges.</p>
<p><b>Working explanation:</b> part of the Mohawk is diverted into the Barge Canal upstream of the Cohoes gauge, and the canal's locks at Waterford return it to the Hudson between the upstream gauges and Green Island. That water reaches Green Island without passing either upstream gauge. A roughly constant diversion is a bigger share of the river at low autumn flows, which matches the pattern. The next test is to line the gap up against the canal's opening and closing dates.</p>
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
<li>The simple mass-balance forecast lost to "no change" on the real river, because of water the gauges cannot see. Reporting that honestly is more useful than hiding it.</li>
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
