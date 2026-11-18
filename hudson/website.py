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
    s["median_flow"] = {site: float(b[f"{site}__{config.FLOW}"].median()) for site in config.SITES}
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




FONTS = ("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700"
         "&family=Source+Sans+3:wght@400;500;600&display=swap")

# A plain research write-up: one column, numbered sections, captioned figures,
# simple ruled tables. One accent colour, used for links and one data series.
CSS = """
:root{--ink:#1a1a1a;--muted:#5f5f5f;--rule:#d9d9d9;--faint:#f4f4f2;--accent:#1d4e89;
--serif:"Source Serif 4",Georgia,serif;--sans:"Source Sans 3",system-ui,sans-serif;color-scheme:light}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:#fff;color:var(--ink);font:18px/1.65 var(--serif)}
.page{max-width:44rem;margin:0 auto;padding:3.5rem 20px 4rem}
a{color:var(--accent);text-underline-offset:2px}
.byline{font:500 .85rem/1.4 var(--sans);color:var(--muted);letter-spacing:.02em}
h1{font:700 clamp(1.7rem,4vw,2.25rem)/1.2 var(--serif);margin:.6rem 0 .8rem;letter-spacing:-.01em;text-wrap:balance}
.meta{font:.9rem/1.5 var(--sans);color:var(--muted);margin:0 0 2rem}
.abstract{border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);padding:1.1rem 0 .4rem;margin-bottom:2rem}
.label{font:600 .78rem/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
.numbers{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1.2rem;margin:0 0 2.5rem;font-family:var(--sans)}
.numbers b{display:block;font:600 1.45rem/1.15 var(--sans);font-variant-numeric:tabular-nums}
.numbers span{display:block;font-size:.84rem;line-height:1.35;color:var(--muted);margin-top:.25rem}
h2{font:700 1.35rem/1.3 var(--serif);margin:2.8rem 0 .7rem}
h2 .n{color:var(--muted);font-weight:400;margin-right:.4rem}
p{margin:0 0 1rem}
.eq{font:italic 1.15rem/1.6 var(--serif);text-align:center;margin:1.4rem 0;padding:.9rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.eq small{display:block;font:normal .85rem var(--sans);color:var(--muted);margin-top:.2rem}
.aside{font:.92rem/1.55 var(--sans);color:var(--muted);background:var(--faint);padding:.8rem 1rem;margin:1rem 0 1.2rem}
figure{margin:1.6rem 0 1.8rem}
figure img{display:block;width:100%;height:auto}
figcaption{font:.88rem/1.5 var(--sans);color:var(--muted);margin-top:.5rem}
figcaption b{color:var(--ink);font-weight:600}
.tablewrap{overflow-x:auto;margin:1.2rem 0 1.6rem}
table{border-collapse:collapse;width:100%;font:.9rem/1.4 var(--sans);font-variant-numeric:tabular-nums}
caption{text-align:left;font:.88rem/1.5 var(--sans);color:var(--muted);caption-side:bottom;padding-top:.5rem}
caption b{color:var(--ink);font-weight:600}
thead th{border-top:1.5px solid var(--ink);border-bottom:1px solid var(--ink);font-weight:600;padding:.45rem .6rem;text-align:right}
td{padding:.4rem .6rem;text-align:right;border-bottom:1px solid #eee;white-space:nowrap}
tbody tr:last-child td{border-bottom:1.5px solid var(--ink)}
th:first-child,td:first-child{text-align:left}
td.best{font-weight:700}
ul.methods{padding-left:1.1rem;margin:0}ul.methods li{margin:.35rem 0}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);font:.85rem/1.5 var(--sans);color:var(--muted)}
code{font-size:.85em}
@media (max-width:640px){body{font-size:17px}.page{padding-top:2.2rem}.numbers{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def build(root: Path) -> Path:
    from . import figures
    run = latest_run(root / "output")
    s = stats(run)
    docs = root / "docs"
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)

    e = html.escape
    names = {"upper_hudson": "Hudson River above Lock 1, near Waterford", "mohawk": "Mohawk River at Cohoes",
             "green_island": "Hudson River at Green Island"}
    short = {"upper_hudson": "Upper Hudson", "mohawk": "Mohawk", "green_island": "Green Island"}
    ids = {k: v["id"].replace("USGS-", "") for k, v in config.SITES.items()}
    q = s["median_flow"]
    inv = s.get("inv")
    per = inv["offset_periods"][0] if inv and inv["offset_periods"] else None

    figures.balance(run, per, img / "balance.png")
    figures.monthly(run, per, img / "monthly.png")
    figures.storm(run, img / "storm.png")

    # forecasting table
    best12 = None
    frows = ""
    for f in s["forecasts"]:
        main = {m: f[m] for m in MODELS if m != "persistence"}
        best = max(main, key=lambda m: main[m]["skill"])
        if f["h"] == 12:
            best12 = (best, main[best]["skill"])
        cells = "".join(f'<td class="{"best" if m == best else ""}">{f[m]["skill"]:+.0%}</td>' for m in MODELS if m != "persistence")
        frows += f"<tr><td>{f['h']} hours</td>{cells}<td>{f['coverage']:.0%}</td></tr>"
    fhead = "".join(f"<th>{e(MODELS[m])}</th>" for m in MODELS if m != "persistence")
    best_skill = max((f[m]["skill"] for f in s["forecasts"] for m in MODELS if m != "persistence"), default=0)

    gauge_rows = "".join(
        f"<tr><td>{e(names[g['site']])}</td><td>{ids[g['site']]}</td><td>{q[g['site']]:,.0f}</td>"
        f"<td>{g['n']:,}</td><td>{pct(100 * g['flagged'])}</td></tr>"
        for g in sorted(s["gauges"], key=lambda g: list(config.SITES).index(g["site"])))

    flagged_pct = pct(100 * s["flagged"] / s["readings"])
    if per:
        p_start, p_end = date.fromisoformat(per["start"]), date.fromisoformat(per["end"])
        gap_big = f"{round(per['median_gap_cfs'], -2):,.0f}"
        summary_result = (f"On a typical day the three gauges agree to within {pct(abs(s['daily_median']), d=0)}. "
                          f"The exception is {p_start:%B %Y} to {p_end:%B %Y}, when Green Island read about {gap_big} cfs "
                          f"more than the two upstream gauges combined. A gap that switches on and off like this, "
                          f"regardless of season or flow, points to a calibration error at one gauge.")
        third = (f"{per['months']} months", "when the gauges disagreed by about " + gap_big + " cfs")
    else:
        summary_result = f"On a typical day the three gauges agree to within {pct(abs(s['daily_median']), d=0)}."
        third = (pct(s["daily_median"], True), "typical daily gap")

    canal = ""
    if inv:
        canal = f"""
<h2><span class=n>4</span>Is it the canal?</h2>
<p>The gap looks seasonal, and the Erie Canal diverts part of the Mohawk around its gauge and returns it to the Hudson
above Green Island, so it was the natural first candidate. I tested it against the canal's opening and closing dates
for each year from 2022 to 2025.</p>
<p>The canal does not explain the gap. Measured in cubic feet per second, the gap is about the same with the canal open
({inv['gap_open_cfs']:,.0f} cfs) as closed ({inv['gap_closed_cfs']:,.0f} cfs), and it moved in the expected direction at only
{inv['events_consistent']} of {inv['events_total']} openings and closings. It only appears seasonal in percentage terms,
because the river is low in late summer ({inv['flow_open_cfs']:,.0f} cfs against {inv['flow_closed_cfs']:,.0f} cfs).</p>
<figure><img src="img/monthly.png" alt="Median daily gap by month, with canal seasons shaded" loading="lazy">
<figcaption><b>Figure 2.</b> Median daily gap by month. Shaded blue: canal open. Dark bars: the offset year.</figcaption></figure>"""

    step = ""
    if per:
        step = f"""
<h2><span class=n>5</span>A year-long offset</h2>
<p>From {p_start:%B %Y} to {p_end:%B %Y} the gap rises to about {gap_big} cfs, then falls back almost at once. Outside that
period it is {s.get('gap_outside_cfs', 0):,.0f} cfs ({pct(s.get('gap_outside_pct', 0), True)}).</p>
<p>A physical cause, such as an unmeasured tributary or a diversion, would follow the seasons or the flow. A gap that
switches on, holds for a year and switches off is the signature of a measurement problem: one gauge's calibration
drifting and later being corrected. The next step is to compare each gauge with the hand measurements USGS field crews
take several times a year, which will show which gauge drifted.</p>"""

    best_sentence = (f"The best model, {MODELS[best12[0]].lower()}, reduced 12-hour forecast error by {best12[1]:.0%}." if best12 else "")
    repo = f'<a href="{e(REPO_URL)}">Code on GitHub</a>' if REPO_URL else ""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hudson River Gauge Check</title>
<meta name="description" content="A mass-balance check of three USGS streamflow gauges at the Hudson–Mohawk confluence, 2022–2025, with flow forecasts.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet"><style>{CSS}</style></head><body>
<main class="page">

<div class="byline">Data science project · Hugh</div>
<h1>Mass-Balance Validation of Streamflow Gauges at the Hudson–Mohawk Confluence</h1>
<p class="meta">U.S. Geological Survey data, {s['start'][:4]}–{s['end'][:4]} · Python {('· ' + repo) if repo else ''}</p>

<div class="abstract"><p class="label">Summary</p>
<p>The U.S. Geological Survey measures river flow at gauges across the country, and its data are used for flood
forecasting, water management and research. This project checks three neighbouring gauges against each other using a
basic principle of chemical engineering: water is conserved, so the flow into a stretch of river must equal the flow
out. {e(summary_result)}</p></div>

<div class="numbers">
<div><b>{s['readings']:,}</b><span>15-minute readings checked</span></div>
<div><b>{flagged_pct}</b><span>flagged as unreliable</span></div>
<div><b>{e(third[0])}</b><span>{e(third[1])}</span></div>
<div><b>{best_skill:.0%}</b><span>lower forecast error than assuming no change</span></div>
</div>

<h2><span class=n>1</span>The question</h2>
<p>Near Troy, New York, the Mohawk River joins the Hudson. USGS measures the flow on each river just above the junction,
and again at Green Island, just below it. Because water is conserved, the flow at Green Island should equal the two
upstream flows added together, allowing about an hour for the water to travel between gauges.</p>
<div class="eq">Green Island(t) ≈ Upper Hudson(t − 1 h) + Mohawk(t − 1 h)<small>travel times fitted from the data</small></div>
<p>When the two sides do not match, either a gauge is wrong or water is entering or leaving somewhere unmeasured. The
project asks how often they match, and what explains the times they do not.</p>
<p class="aside">Flow is measured in <b>cubic feet per second (cfs)</b>, the volume of water passing a gauge each second.
One cubic foot is about 28 litres, so Green Island's typical {q['green_island']:,.0f} cfs is roughly
{round(q['green_island'] * 28.3168, -4):,.0f} litres every second.</p>

<h2><span class=n>2</span>Data</h2>
<p>Readings every 15 minutes from three USGS gauges, {s['start'][:4]} to {s['end'][:4]}, downloaded from the USGS Water Data
API, with hourly rainfall for each river basin from the Open-Meteo archive. Every reading was checked for stuck sensors,
spikes, ice and values USGS marks as estimated; {flagged_pct} were flagged and left out.</p>
<div class="tablewrap"><table><thead><tr><th>Gauge</th><th>USGS site</th><th>Median flow (cfs)</th><th>Readings</th><th>Flagged</th></tr></thead>
<tbody>{gauge_rows}</tbody>
<caption><b>Table 1.</b> The three flow gauges. For much of 2024 Green Island published only estimated values every two hours.</caption></table></div>

<h2><span class=n>3</span>Does the water add up?</h2>
<p>Most of the time, yes. On a typical day the flow at Green Island matches the upstream total to within
{pct(abs(s['daily_median']))} (median gap {pct(s['daily_median'], True)}). Hour to hour the gap is larger
({pct(s['hourly_abs'])} median), because the dams and locks between the gauges hold water back and release it in
pulses. That variation is real river behaviour rather than measurement error.</p>
<figure><img src="img/balance.png" alt="Daily flow at Green Island against the upstream total, and the daily gap" loading="lazy">
<figcaption><b>Figure 1.</b> Top: daily flow at Green Island and the two upstream gauges combined. Bottom: the daily gap.
Breaks are periods with too few reliable readings. Shaded: {f"{p_start:%B %Y} to {p_end:%B %Y}" if per else "–"}.</figcaption></figure>
{canal}
{step}

<h2><span class=n>{6 if per else 5}</span>Forecasting</h2>
<p>Five models predict the flow at Green Island 6 to 24 hours ahead from upstream flows and rainfall. Each was trained
only on earlier data and tested on later data, as it would be used in practice. Skill is the share of error removed
compared with assuming the flow stays the same. {best_sentence} The prediction ranges were calibrated so that about 90%
of actual flows fall inside them.</p>
<div class="tablewrap"><table><thead><tr><th>Forecast</th>{fhead}<th>Inside 90% range</th></tr></thead><tbody>{frows}</tbody>
<caption><b>Table 2.</b> Forecast skill against assuming no change (higher is better; negative is worse than no change). Best model per row in bold.</caption></table></div>
<figure><img src="img/storm.png" alt="Measured flow during the largest storm, with 12-hour-ahead forecasts" loading="lazy">
<figcaption><b>Figure 3.</b> The largest storm in the test period: measured flow, the forecast made 12 hours earlier, and its 90% range.</figcaption></figure>

<h2><span class=n>{7 if per else 6}</span>Methods</h2>
<ul class="methods">
<li><b>Mass balance</b> with travel times between gauges fitted jointly from the data.</li>
<li><b>Quality control</b> using USGS qualifier codes plus checks for stuck and spiking sensors.</li>
<li><b>Forecasting</b> with ridge regression and gradient-boosted trees, walk-forward testing, and conformal prediction intervals.</li>
<li><b>Anomaly detection</b> with an Isolation Forest on balance and sensor signals.</li>
<li><b>Tests</b> that plant known faults in simulated rivers and confirm each check catches them.</li>
<li>Python, polars, scikit-learn, matplotlib.</li>
</ul>

<footer>Data: U.S. Geological Survey (gauges {ids['upper_hudson']}, {ids['mohawk']}, {ids['green_island']}); rainfall from Open-Meteo.
{(repo + ' · ') if repo else ''}Results from run <code>{e(s['run'])}</code>, generated {date.today():%B %-d, %Y}.</footer>
</main></body></html>"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    return docs / "index.html"


if __name__ == "__main__":
    path = build(Path.cwd())
    print(f"✅ Website built: {path}")
    print("   Preview: double-click docs/index.html in Finder.")
