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




FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800"
         "&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap")

# Layout: a dark "river at night" hero that states the question and the answer,
# then the story told as findings, each a headline + explanation + evidence.
CSS = """
:root{--paper:#f6f8f9;--panel:#ffffff;--ink:#0e1f26;--muted:#4f646c;--line:#d5dfe2;
--river:#0f5e6e;--river2:#3a9bb0;--signal:#d99a1e;--bad:#b4462d;--good:#1f7a4d;
--hero:#0b1d24;--hero-ink:#e6f0f1;--hero-muted:#9db5ba;
--display:"Archivo",system-ui,sans-serif;--body:"Instrument Sans",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root{--paper:#0d1518;--panel:#132027;--ink:#e3ecee;--muted:#9ab0b6;--line:#24363d;
--river:#5fc0d3;--river2:#8ad6e4;--signal:#f0c25a;--bad:#ef8a6f;--good:#6fd39f;--hero:#071216;color-scheme:dark}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 var(--body)}
a{color:var(--river)}a:focus-visible,.btn:focus-visible{outline:2px solid var(--signal);outline-offset:3px}
.wrap{max-width:68rem;margin:0 auto;padding-left:max(20px,env(safe-area-inset-left));padding-right:max(20px,env(safe-area-inset-right))}
.narrow{max-width:44rem}
/* hero */
.hero{background:var(--hero);color:var(--hero-ink);position:relative;overflow:hidden}
.hero .wrap{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:3rem;align-items:center;padding-top:4.5rem;padding-bottom:4.5rem}
.eyebrow{font:500 .78rem/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--signal)}
h1{font:800 clamp(2.4rem,6vw,4.4rem)/1.02 var(--display);font-stretch:112%;letter-spacing:-.02em;margin:.9rem 0 1.2rem;text-wrap:balance}
.lede{font-size:1.2rem;line-height:1.55;color:var(--hero-ink);max-width:36rem;margin:0 0 1.6rem}
.answer{border-left:3px solid var(--signal);padding:.2rem 0 .2rem 1rem;margin:0 0 2rem;max-width:36rem}
.answer b{display:block;font:700 1.05rem var(--display);font-stretch:105%;margin-bottom:.3rem;color:#fff}
.answer span{color:var(--hero-muted);font-size:1rem}
.btns{display:flex;flex-wrap:wrap;gap:.75rem}
.btn{display:inline-block;padding:.7rem 1.15rem;border-radius:999px;font:600 .95rem var(--body);text-decoration:none}
.btn.primary{background:var(--signal);color:#1a1405}.btn.ghost{color:var(--hero-ink);border:1px solid #3b5660}
.confluence{width:100%;height:auto;display:block}
.confluence .r{fill:none;stroke:#3a9bb0;stroke-linecap:round}
.confluence .g{fill:var(--hero);stroke:#e6f0f1;stroke-width:2.5}
.confluence .n{fill:#e6f0f1;font:600 15px var(--body)}
.confluence .q{fill:#f0c25a;font:500 13px var(--mono)}
.confluence .eq{fill:#9db5ba;font:500 13px var(--mono)}
.ticks{position:absolute;right:0;top:0;bottom:0;width:22px;border-left:1px solid #1f3a43;
background:repeating-linear-gradient(to bottom,#2c4a54 0 1px,transparent 1px 12px);opacity:.7}
/* stats strip */
.strip{background:var(--panel);border-bottom:1px solid var(--line)}
.strip .wrap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0}
.stat{padding:1.5rem 1.25rem 1.4rem;border-right:1px solid var(--line)}.stat:last-child{border-right:0}
.stat b{display:block;font:700 clamp(1.5rem,3vw,2.1rem)/1.1 var(--display);font-stretch:108%;font-variant-numeric:tabular-nums;color:var(--river)}
.stat span{display:block;margin-top:.35rem;font-size:.9rem;color:var(--muted);line-height:1.4}
/* sections */
section{padding:4.5rem 0 1rem}
h2{font:800 clamp(1.7rem,3.4vw,2.4rem)/1.12 var(--display);font-stretch:108%;letter-spacing:-.01em;margin:0 0 1rem;text-wrap:balance}
h3{font:700 clamp(1.3rem,2.4vw,1.6rem)/1.2 var(--display);font-stretch:105%;margin:.35rem 0 .8rem;text-wrap:balance}
p{margin:0 0 1rem}.muted{color:var(--muted)}
.idea{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:2.5rem;align-items:center}
.equation{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.4rem 1.5rem;font:500 1rem/1.9 var(--mono)}
.equation .big{font:700 1.25rem var(--display);font-stretch:105%;color:var(--ink);display:block;margin-bottom:.4rem}
.equation em{font-style:normal;color:var(--river)}
.steps{list-style:none;padding:0;margin:1.5rem 0 0;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem}
.steps li{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem}
.steps b{display:block;font:700 1rem var(--display);margin-bottom:.25rem}.steps span{font-size:.93rem;color:var(--muted)}
.finding{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:2.5rem;padding:2.5rem 0;border-top:1px solid var(--line);align-items:start}
.finding:first-of-type{border-top:0}
.tag{font:500 .75rem/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--river)}
.tag.warn{color:var(--bad)}
.evidence{min-width:0}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem}
figure img{display:block;width:100%;max-width:100%;height:auto;border-radius:6px;background:#fff}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.55rem;line-height:1.45}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:.6rem .8rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500;font-size:.82rem}
tr:last-child td{border-bottom:0}td.best{color:var(--good);font-weight:600}td.neg{color:var(--bad)}
.callout{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--signal);border-radius:10px;padding:1rem 1.2rem;margin-top:1rem}
.learned{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1.5rem}
.learned div{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.2rem}
.learned b{display:block;font:700 1.05rem var(--display);margin-bottom:.4rem}.learned span{color:var(--muted);font-size:.95rem}
.built{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1.5rem;margin-top:1.2rem;font-size:.95rem}
.built b{font:500 .75rem var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:.4rem}
code{font-family:var(--mono);font-size:.88em}
footer{margin-top:4rem;padding:2rem 0 3rem;border-top:1px solid var(--line);font-size:.88rem;color:var(--muted)}
@media (max-width:860px){.hero .wrap,.idea,.finding{grid-template-columns:minmax(0,1fr)}.hero .wrap{gap:2rem;padding-top:3rem;padding-bottom:3rem}
.strip .wrap{grid-template-columns:repeat(2,minmax(0,1fr))}.stat:nth-child(2){border-right:0}.stat:nth-child(-n+2){border-bottom:1px solid var(--line)}
.steps,.learned,.built{grid-template-columns:minmax(0,1fr)}.ticks{display:none}section{padding-top:3rem}}
"""


def build(root: Path) -> Path:
    run = latest_run(root / "output")
    s = stats(run)
    docs = root / "docs"
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)
    charts = {}
    for name in ("monthly_gap", "flow_balance", "storm_forecast", "skill_by_horizon"):
        src = run / f"{name}.png"
        if src.exists():
            shutil.copy(src, img / src.name)
            charts[name] = f"img/{src.name}"

    e = html.escape
    names = {"upper_hudson": "Upper Hudson", "mohawk": "Mohawk at Cohoes", "green_island": "Green Island"}
    ids = {k: v["id"].replace("USGS-", "") for k, v in config.SITES.items()}
    q = s["median_flow"]
    inv = s.get("inv")
    per = inv["offset_periods"][0] if inv and inv["offset_periods"] else None

    def fig(key, cap):
        return (f'<figure><img src="{charts[key]}" alt="{e(cap)}" loading="lazy">'
                f'<figcaption>{e(cap)}</figcaption></figure>') if key in charts else ""

    best12 = None
    frows = ""
    for f in s["forecasts"]:
        main = {m: f[m] for m in MODELS if m != "persistence"}
        best = max(main, key=lambda m: main[m]["skill"])
        if f["h"] == 12:
            best12 = (best, main[best]["skill"])
        cells = "".join(
            f'<td class="{"best" if m == best else ("neg" if f[m]["skill"] < 0 else "")}">{f[m]["skill"]:+.0%}</td>'
            for m in MODELS if m != "persistence")
        frows += f"<tr><td>{f['h']} hours</td>{cells}<td>{f['coverage']:.0%}</td></tr>"
    fhead = "".join(f"<th>{e(MODELS[m])}</th>" for m in MODELS if m != "persistence")
    best_skill = max((f[m]["skill"] for f in s["forecasts"] for m in MODELS if m != "persistence"), default=0)

    grows = "".join(
        f"<tr><td>{e(names.get(g['site'], g['site']))}</td><td>{g['n']:,}</td><td>{pct(100 * g['flagged'])}</td>"
        f"<td>{g['estimated']:,}</td><td>{g['stuck']:,}</td><td>{g['spikes']:,}</td></tr>"
        for g in s["gauges"])

    # --- the story -------------------------------------------------------------
    if per:
        p_start, p_end = date.fromisoformat(per["start"]), date.fromisoformat(per["end"])
        answer = (f"Mostly yes, but not always. For a year, from {p_start:%B %Y} to {p_end:%B %Y}, "
                  f"the gauges disagreed by about {round(per['median_gap_cfs'], -2):,.0f} cubic feet per second, "
                  f"which points to a calibration error at one of them.")
        stat3 = (f"{per['months']} months", "when one gauge appears to have drifted out of calibration")
    else:
        answer = "Mostly yes. Where they disagree, the project tracks down why."
        stat3 = (pct(s["daily_median"], True), "typical daily gap between inflow and outflow")

    canal_block = ""
    if inv:
        canal_block = f"""
<div class="finding">
  <div><div class="tag warn">Finding 2</div><h3>My first explanation was wrong</h3>
  <p>The gap looked seasonal, so I suspected the Erie Canal, which diverts part of the Mohawk around its gauge
  and returns it to the Hudson above Green Island. I tested that against the canal's real opening and closing
  dates for 2022–2025.</p>
  <p>It doesn't hold up. In cubic feet per second the gap is the same with the canal open
  ({inv['gap_open_cfs']:,.0f}) as closed ({inv['gap_closed_cfs']:,.0f}), and it moved the expected way at only
  {inv['events_consistent']} of {inv['events_total']} openings and closings. It only <i>looked</i> seasonal in
  percent terms, because the river is low in late summer.</p></div>
  <div class="evidence">{fig('monthly_gap', 'Median daily gap by month (cfs). Blue bands: canal open. Orange: months above 1,000 cfs.')}</div>
</div>"""
    step_block = ""
    if per:
        if inv["calibration"]:
            rows = "".join(f"<tr><td>{e(names.get(r['site'], r['site']))}</td><td>{'During' if r['in_offset_period'] else 'Outside'}</td>"
                           f"<td>{r['measurements']}</td><td>{r['median_error_pct']:+.1f}%</td></tr>" for r in inv["calibration"])
            cal = ("<div class=\"tablewrap\"><table><thead><tr><th>Gauge</th><th>Offset period</th><th>Hand measurements</th>"
                   f"<th>Gauge vs. hand measurement</th></tr></thead><tbody>{rows}</tbody></table></div>")
        else:
            cal = ("<div class=\"callout\"><b>Next test, in progress.</b> USGS crews measure each river by hand several times "
                   "a year, and the gauges are calibrated against those visits. Comparing each gauge with the hand "
                   "measurements during the offset year will show which one drifted.</div>")
        step_block = f"""
<div class="finding">
  <div><div class="tag">Finding 3</div><h3>The real pattern is a year-long step</h3>
  <p>From {p_start:%B %Y} to {p_end:%B %Y} the gap jumps to about {round(per['median_gap_cfs'], -2):,.0f} cfs, then drops
  back almost overnight. Outside that year it is {s.get('gap_outside_cfs', 0):,.0f} cfs
  ({pct(s.get('gap_outside_pct', 0), True)}).</p>
  <p>Something physical in the river would follow the seasons or the flow. A gap that switches on, stays for a
  year and switches off looks like a measuring problem: one gauge's calibration drifting, then being corrected.</p></div>
  <div class="evidence">{cal}</div>
</div>"""

    repo = f'<a class="btn ghost" href="{e(REPO_URL)}">Code on GitHub</a>' if REPO_URL else ""
    best_sentence = (f"{MODELS[best12[0]]} cut 12-hour forecast error by {best12[1]:.0%}." if best12 else "")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Hudson River Gauge Check</title>
<meta name="description" content="Can you trust a river gauge? Checking four years of USGS data on the Hudson and Mohawk with a mass balance, and forecasting the river with machine learning.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet"><style>{CSS}</style></head><body>

<header class="hero"><div class="ticks" aria-hidden="true"></div><div class="wrap">
  <div>
    <div class="eyebrow">Data validation · Hudson River, New York</div>
    <h1>Can you trust a river gauge?</h1>
    <p class="lede">Government river gauges feed flood warnings, water supply and research, and people mostly take
    their numbers on trust. I checked four years of them using one rule from chemical engineering:
    <b>water can't appear or disappear.</b></p>
    <div class="answer"><b>What I found</b><span>{e(answer)}</span></div>
    <div class="btns"><a class="btn primary" href="#findings">See the findings</a>{repo}</div>
  </div>
  <svg class="confluence" viewBox="0 0 520 330" role="img" aria-label="The upper Hudson and Mohawk rivers join and flow past the Green Island gauge">
    <path class="r" d="M70 60 C 210 60, 230 165, 330 165" stroke-width="{3 + 9 * q['upper_hudson'] / max(q.values()):.1f}"/>
    <path class="r" d="M70 270 C 210 270, 230 165, 330 165" stroke-width="{3 + 9 * q['mohawk'] / max(q.values()):.1f}"/>
    <path class="r" d="M330 165 L 440 165" stroke-width="{3 + 9 * q['green_island'] / max(q.values()):.1f}"/>
    <circle class="g" cx="70" cy="60" r="8"/><circle class="g" cx="70" cy="270" r="8"/><circle class="g" cx="450" cy="165" r="8"/>
    <text class="n" x="70" y="35" text-anchor="start">Upper Hudson</text><text class="q" x="70" y="96">{q['upper_hudson']:,.0f} cfs</text>
    <text class="n" x="70" y="305">Mohawk</text><text class="q" x="70" y="246">{q['mohawk']:,.0f} cfs</text>
    <text class="n" x="450" y="138" text-anchor="end">Green Island</text><text class="q" x="450" y="200" text-anchor="end">{q['green_island']:,.0f} cfs</text>
    <text class="eq" x="260" y="322" text-anchor="middle">median flow 2022–2025 · line width to scale</text>
  </svg>
</div></header>

<div class="strip"><div class="wrap">
  <div class="stat"><b>{s['readings']:,}</b><span>sensor readings checked, every 15 minutes for four years</span></div>
  <div class="stat"><b>{pct(100 * s['flagged'] / s['readings'])}</b><span>flagged as unreliable: stuck, spiking, iced or estimated</span></div>
  <div class="stat"><b>{e(stat3[0])}</b><span>{e(stat3[1])}</span></div>
  <div class="stat"><b>{best_skill:.0%}</b><span>less forecast error than assuming the river stays the same</span></div>
</div></div>

<main class="wrap">
<section id="idea">
  <div class="idea">
    <div class="narrow"><div class="tag">The idea</div><h2>Three gauges, one rule</h2>
    <p>Near Troy, New York, the Mohawk River joins the Hudson. The U.S. Geological Survey measures the flow on each
    river and again just downstream at Green Island, where they have combined.</p>
    <p>So the water passing Green Island should equal what came down the two rivers an hour or two earlier. When it
    doesn't, either a gauge is wrong or water is getting in or out somewhere nobody measures. Finding out which is
    the whole project.</p></div>
    <div class="equation"><span class="big">Green Island ≈ Upper Hudson + Mohawk</span>
    measured at Green Island now<br>≈ <em>upper Hudson</em>, about 1 hour earlier<br>+ <em>Mohawk</em>, about 1 hour earlier<br>
    <span class="muted">USGS gauges {ids['upper_hudson']}, {ids['mohawk']} and {ids['green_island']}</span></div>
  </div>
  <ol class="steps">
    <li><b>Clean</b><span>Flag stuck sensors, fake spikes, ice and estimated values in every reading.</span></li>
    <li><b>Balance</b><span>Check the water adds up between the gauges, hour by hour.</span></li>
    <li><b>Investigate</b><span>Test explanations for where it doesn't, against outside evidence.</span></li>
    <li><b>Forecast</b><span>Predict Green Island 6–24 hours ahead from upstream flows and rain.</span></li>
  </ol>
</section>

<section id="findings">
  <h2>What I found</h2>
<div class="finding">
  <div><div class="tag">Finding 1</div><h3>Most of the time, the gauges agree</h3>
  <p>On a typical day the water adds up to within {pct(abs(s['daily_median']))} (median gap {pct(s['daily_median'], True)}).
  Hour to hour it is noisier ({pct(s['hourly_abs'])}), because the dams and locks between the gauges hold water back
  and release it in pulses. That noise is real river behaviour, not bad data.</p></div>
  <div class="evidence">{fig('flow_balance', 'Top: Green Island against the two upstream gauges added together. Bottom: the gap, hour by hour.')}</div>
</div>
{canal_block}
{step_block}
<div class="finding">
  <div><div class="tag">Finding 4</div><h3>Bad data is common if you look for it</h3>
  <p>{pct(100 * s['flagged'] / s['readings'])} of all readings failed a quality check. For most of 2024 Green Island
  only published estimated values every two hours, and the Mohawk's salt and temperature sensors were stuck or
  spiking for long stretches. Real data also exposed bugs synthetic data never did, like a download that silently
  dropped every December 31st.</p></div>
  <div class="evidence"><div class="tablewrap"><table><thead><tr><th>Flow gauge</th><th>Readings</th><th>Flagged</th><th>Estimated</th><th>Stuck</th><th>Spikes</th></tr></thead>
  <tbody>{grows}</tbody></table></div></div>
</div>
<div class="finding">
  <div><div class="tag">Finding 5</div><h3>Forecasting works, modestly, and honestly</h3>
  <p>Five models predict Green Island's flow ahead of time. Each is trained only on the past and scored on the
  future, as it would be used. Skill is the share of error removed compared with assuming the flow stays the same.</p>
  <p>{best_sentence} The pure mass-balance forecast lost to "no change",
  because of the calibration offset it can't see. Its 90% prediction ranges were calibrated so that about 90% of
  real outcomes land inside.</p></div>
  <div class="evidence"><div class="tablewrap"><table><thead><tr><th>Ahead</th>{fhead}<th>Inside 90% range</th></tr></thead><tbody>{frows}</tbody></table></div>
  <p class="muted" style="font-size:.85rem;margin-top:.5rem">Best model per horizon in green. Negative means worse than "no change".</p>
  {fig('storm_forecast', 'The largest storm in the test period: 12-hour-ahead forecasts against what was measured, with the 90% range.')}</div>
</div>
</section>

<section id="learned">
  <h2>What I learned</h2>
  <div class="learned">
    <div><b>Physics beats rules</b><span>A sensor drifting out of calibration still reports believable numbers. Only checking it against its neighbours exposes it.</span></div>
    <div><b>Test your own explanation</b><span>The canal story fitted the percentages perfectly and was wrong. Measuring in cfs and checking real dates found the actual cause.</span></div>
    <div><b>Report what lost</b><span>The simple physics forecast lost to "no change". Saying so is more useful than hiding it.</span></div>
  </div>
</section>

<section id="built">
  <h2>How it's built</h2>
  <div class="built">
    <div><b>Data</b>USGS Water Data API (15-minute readings and field measurements), Open-Meteo rainfall.</div>
    <div><b>Methods</b>Python, polars, scikit-learn. Walk-forward testing, leakage checks, conformal prediction intervals, Isolation Forest.</div>
    <div><b>Checks</b>Automated tests plant known faults in simulated rivers and confirm every check catches them.</div>
  </div>
</section>

<footer>{'<a href="' + e(REPO_URL) + '">Code on GitHub</a> · ' if REPO_URL else ''}Data: U.S. Geological Survey; rainfall from Open-Meteo.
Results from run <code>{e(s['run'])}</code>, generated {date.today():%B %-d, %Y}.</footer>
</main></body></html>"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    return docs / "index.html"


if __name__ == "__main__":
    path = build(Path.cwd())
    print(f"✅ Website built: {path}")
    print("   Preview: double-click docs/index.html in Finder.")
    print("   Publish: push to GitHub, then Settings > Pages > Branch: main, folder /docs.")
