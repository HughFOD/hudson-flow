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




FONTS = ("https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700"
         "&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&display=swap")

# Shares its design system with the Gas Storage Nowcast page (hughod.com/gas-storage-nowcast):
# same tokens, type, masthead, centrepiece card, score tiles and theme-aware SVG charts.
CSS = """
:root{--paper:#eef1f4;--card:#fbfcfd;--ink:#13202e;--muted:#5d6975;--faint:#8996a3;--line:#e2e6ea;
--flame:#1d5fd1;--flame-soft:#dbe6f8;--heat:#c8541f;--heat-soft:#f6e1d6;--grid:#e9edf0;--link:#3a66a8;
--display:"Barlow Semi Condensed","Arial Narrow",sans-serif;--body:"Source Sans 3",-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root{--paper:#0c131b;--card:#141e28;--ink:#e6ecf2;--muted:#94a2b0;--faint:#6b7987;
--line:#1f2a36;--flame:#4f86e8;--flame-soft:#1d3254;--heat:#e0703a;--heat-soft:#4a2a1b;--grid:#1b2631;--link:#8eb0e6}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 var(--body)}
a{color:var(--link);text-decoration:none;text-underline-offset:3px}a:hover{text-decoration:underline}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px}
.mast{padding:22px 0 0}.mast .row{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.mast .kicker,.mast .brand,.mast a{white-space:nowrap}
@media (max-width:560px){.mast .kicker{font-size:.72rem;letter-spacing:.09em}.mast .brand{font-size:.95rem;letter-spacing:.1em}.mast .wide{display:none}}
.mast .row:first-child{padding-bottom:10px;border-bottom:2px solid var(--ink)}
.mast .row+.row{padding-top:8px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.kicker{font-family:var(--display);font-weight:600;font-size:.82rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.mast .brand{font-family:var(--display);font-weight:700;font-size:1.05rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink)}
.mast .tail{display:flex;gap:22px;align-items:baseline}
h1,h2,h3{font-family:var(--display);font-weight:600;letter-spacing:-.005em}
h1{font-size:clamp(2.1rem,4.6vw,3.3rem);line-height:1.05;margin:40px 0 14px;max-width:15em;text-wrap:balance}
h2{font-size:1.75rem;line-height:1.15;margin:0 0 6px}h3{font-size:1.2rem;margin:0 0 4px}
.lede{font-size:1.08rem;color:var(--muted);max-width:40em;margin:0 0 30px}
section{padding:80px 0 0}.sub{color:var(--muted);max-width:44em;margin:0 0 22px;font-size:.97rem}
section>.kicker{display:block;margin-bottom:6px;color:var(--flame)}
.call{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 30px 24px;box-shadow:0 1px 0 var(--line),0 18px 40px -28px rgba(20,40,70,.35)}
.call-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.pill{display:inline-block;font-family:var(--display);font-weight:700;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--flame);background:var(--flame-soft);padding:4px 10px;border-radius:999px}
.call .when{color:var(--muted);margin:0;font-size:.95rem}.call .when b{color:var(--ink);font-weight:600}
.call-main{display:grid;grid-template-columns:1fr auto;gap:18px 48px;align-items:end;margin:14px 0 6px}
.figline{display:flex;align-items:baseline;gap:12px}
.call .fig{font-family:var(--display);font-weight:700;font-size:clamp(4.6rem,13vw,8.4rem);line-height:.82;color:var(--flame);font-variant-numeric:tabular-nums;letter-spacing:-.025em}
.call .unit{font-family:var(--display);font-size:2rem;font-weight:600;color:var(--flame);opacity:.85}
.call .what{font-size:1.1rem;color:var(--ink);margin:12px 0 0}
.side{display:grid;gap:14px;min-width:220px;padding-left:28px;border-left:1px solid var(--line)}
.side .kicker{display:block;margin-bottom:2px}
.side .val{font-family:var(--display);font-weight:600;font-size:1.75rem;line-height:1.1;font-variant-numeric:tabular-nums;white-space:nowrap}
.side .val.avg{color:var(--heat)}.side .val small{font-size:1rem;font-weight:500;color:var(--muted)}
.scale{margin:22px 0 4px;min-width:0}.scale svg{display:block;overflow:visible;max-width:100%}
.keys{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:.86rem;color:var(--muted);margin:2px 0 0}
.keys span{display:inline-flex;align-items:center;gap:7px}
.keys .k-pred{width:11px;height:11px;border-radius:50%;background:var(--flame);box-shadow:0 0 0 3px var(--flame-soft)}
.keys .k-range{width:22px;height:10px;border-radius:3px;background:var(--flame-soft);border:1px solid var(--flame)}
.keys .k-avg{width:3px;height:14px;border-radius:1px;background:var(--heat)}
.keys .k-zero{width:0;height:14px;border-left:2px solid var(--muted)}
.notes{display:grid;grid-template-columns:1fr 1fr;gap:6px 28px;margin:18px 0 0;padding-top:16px;border-top:1px solid var(--line)}
.notes p{margin:0;color:var(--muted);font-size:.92rem}.notes b{color:var(--ink);font-weight:600}
@media (max-width:700px){.call{padding:22px 20px}.call-main{grid-template-columns:1fr}
.side{border-left:0;padding-left:0;border-top:1px solid var(--line);padding-top:14px;grid-template-columns:1fr 1fr}.notes{grid-template-columns:1fr}.side .val{font-size:1.4rem}}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.98rem}
th,td{padding:10px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:0}
th{font-weight:600;color:var(--muted);font-size:.9rem;vertical-align:bottom;white-space:normal}
th:first-child,td:first-child{text-align:left}
.missbar{display:inline-block;height:8px;border-radius:0 4px 4px 0;background:var(--flame);vertical-align:middle;margin-left:8px}
.missbar.worse{background:var(--heat)}
.win{color:var(--flame);font-weight:600}.lose{color:var(--heat);font-weight:600}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:0;background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:18px}
.score{padding:18px 22px;border-left:1px solid var(--line)}.score:first-child{border-left:0}
.score .kicker{display:block;margin-bottom:8px}
.score .v{font-family:var(--display);font-weight:700;font-size:2.4rem;line-height:1;font-variant-numeric:tabular-nums}
.score .v small{font-size:1.1rem;font-weight:500;color:var(--muted)}.score .l{color:var(--muted);font-size:.93rem;margin-top:6px}
h3.minor{font-size:1.25rem;margin:34px 0 4px}
@media (max-width:700px){.score{border-left:0;border-top:1px solid var(--line)}.score:first-child{border-top:0}}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 18px 10px;margin:0 0 18px}
.chart h3{margin:0 4px 2px}.chart p{margin:0 4px 8px;color:var(--muted);font-size:.93rem}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;margin:0 4px 6px;font-size:.88rem;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{display:inline-block;width:18px;height:0;border-top:2px solid}
.legend i.band{height:10px;border:0;border-radius:2px}
.plot svg{display:block;width:100%;height:auto;overflow:visible}.plot text{font:12px var(--body);fill:var(--faint)}
.plot.m text{font-size:21px}.scale.m text{font-size:23px}
.m{display:none}@media (max-width:700px){.d{display:none}.m{display:block}}
.steps{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:26px}
.steps li{counter-increment:s;border-top:2px solid var(--ink);padding-top:12px}
.steps li::before{content:counter(s);font-family:var(--display);font-weight:700;font-size:1.5rem;color:var(--flame);display:block;line-height:1;margin-bottom:8px}
.steps p{margin:4px 0 0;color:var(--muted);font-size:.97rem}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px 34px;margin:0}
.findings p{margin:2px 0 0;color:var(--muted)}
.glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px 30px;margin:0}
.glossary dt{font-family:var(--display);font-weight:600;font-size:1.1rem}.glossary dd{margin:2px 0 0;color:var(--muted);font-size:.95rem}
footer{margin:72px 0 0;padding:20px 0 40px;border-top:1px solid var(--line);color:var(--muted);font-size:.9rem}
"""


# ---------- theme-aware SVG charts (colours come from the CSS variables) ----------

def _fmt(v):
    return f"{v / 1000:,.0f}k" if abs(v) >= 1000 else f"{v:,.0f}"


def svg_scale(lo, hi, point, base, xmin, xmax, W=1000) -> str:
    L, R = 30, 30
    x = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - R)
    y = 46
    t = lambda xx, yy, s, anchor="middle", cls="", w="": f'<text x="{xx:.0f}" y="{yy}" text-anchor="{anchor}" style="{w}" class="{cls}">{s}</text>'
    return f"""<svg viewBox="0 0 {W} 92" role="img" aria-label="Offset-year gap compared with zero and with the rest of the period">
<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" stroke="var(--line)" stroke-width="2"/>
<line x1="{x(0):.0f}" x2="{x(0):.0f}" y1="{y - 18}" y2="{y + 18}" stroke="var(--muted)" stroke-width="2"/>
{t(x(0), y + 36, "0", w="fill:var(--muted)")}
<rect x="{x(lo):.0f}" y="{y - 10}" width="{x(hi) - x(lo):.0f}" height="20" rx="5" fill="var(--flame-soft)" stroke="var(--flame)" stroke-width="1.2"/>
{t(x(lo), y + 36, f"+{lo:,.0f}", w="fill:var(--ink);font-weight:600")}{t(x(hi), y + 36, f"+{hi:,.0f}", w="fill:var(--ink);font-weight:600")}
<line x1="{x(base):.0f}" x2="{x(base):.0f}" y1="{y - 16}" y2="{y + 16}" stroke="var(--heat)" stroke-width="3"/>
{t(x(base), y - 24, f"+{base:,.0f}", w="fill:var(--heat);font-weight:600")}
<circle cx="{x(point):.0f}" cy="{y}" r="11" fill="var(--flame-soft)"/><circle cx="{x(point):.0f}" cy="{y}" r="7" fill="var(--flame)"/>
{t(x(point), y - 24, f"+{point:,.0f}", w="fill:var(--flame);font-weight:700;font-size:14px")}
</svg>"""


def svg_bars(months, values, highlight, bands, H=260, W=1000) -> str:
    L, R, T, B = (52, 10, 14, 30) if W >= 800 else (58, 6, 14, 34)
    vmax, vmin = max(max(values), 500) * 1.1, min(min(values), -500) * 1.1
    n = len(months)
    t0, t1 = months[0].toordinal(), months[-1].toordinal() + 31
    x = lambda d: L + (d.toordinal() - t0) / (t1 - t0) * (W - L - R)
    y = lambda v: T + (vmax - v) / (vmax - vmin) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for a, b, colour in bands:
        out.append(f'<rect x="{x(a):.1f}" y="{T}" width="{x(b) - x(a):.1f}" height="{H - T - B}" fill="{colour}" opacity=".6"/>')
    step = 1000 if vmax - vmin < 9000 else 2000
    for g in range(int(vmin // step) * step, int(vmax) + 1, step):
        if vmin <= g <= vmax:
            out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/><text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end">{_fmt(g)}</text>')
    bw = (W - L - R) / (n + 1) * 0.72
    for d, v, h in zip(months, values, highlight):
        y0, y1 = sorted((y(0), y(v)))
        out.append(f'<rect x="{x(d) + 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(y1 - y0, 1):.1f}" rx="2" fill="{"var(--flame)" if h else "var(--faint)"}"/>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--ink)" stroke-width="1.6"/>')
    for yr in sorted({d.year for d in months} | {months[-1].year + 1}):
        xx = x(date(yr, 1, 1))
        if L <= xx <= W - R:
            out.append(f'<text x="{xx:.1f}" y="{H - 8}" text-anchor="middle">{yr}</text>')
    out.append(f'<text x="{L - 44}" y="{T + 4}" style="font-size:11px">cfs</text></svg>')
    return "".join(out)


def svg_track(times, actual, pred, lo, hi, H=280, W=1000) -> str:
    L, R, T, B = (52, 10, 10, 30) if W >= 800 else (58, 6, 10, 34)
    t0, t1 = times[0].timestamp(), times[-1].timestamp()
    vmax = float(np.nanmax(actual)) * 1.3
    x = lambda t: L + (t.timestamp() - t0) / (t1 - t0) * (W - L - R)
    y = lambda v: T + (vmax - v) / vmax * (H - T - B)
    pts = lambda vals: " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(times, vals) if np.isfinite(v))
    band = pts(hi) + " " + " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in reversed(list(zip(times, lo))) if np.isfinite(v))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Actual flow and 12-hour-ahead forecast">']
    for g in range(0, int(vmax) + 1, 10000):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/><text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end">{_fmt(g)}</text>')
    out.append(f'<polygon points="{band}" fill="var(--flame-soft)"/>')
    out.append(f'<polyline points="{pts(actual)}" fill="none" stroke="var(--ink)" stroke-width="1.4"/>')
    out.append(f'<polyline points="{pts(pred)}" fill="none" stroke="var(--flame)" stroke-width="1.6"/>')
    seen = set()
    for t in times:
        if t.weekday() == 0 and t.date() not in seen and t.hour == 0:
            seen.add(t.date())
            if W < 800 and len(seen) % 2 == 0:
                continue
            out.append(f'<text x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:%b %-d}</text>')
    return "".join(out) + "</svg>"


def build(root: Path) -> Path:
    from . import figures
    from .investigate import CANAL_SEASONS
    run = latest_run(root / "output")
    s = stats(run)
    docs = root / "docs"
    e = html.escape
    names = {"upper_hudson": "Hudson above Lock 1, Waterford", "mohawk": "Mohawk at Cohoes", "green_island": "Hudson at Green Island"}
    ids = {k: v["id"].replace("USGS-", "") for k, v in config.SITES.items()}
    q = s["median_flow"]
    inv = s["inv"]
    per = inv["offset_periods"][0]
    p_start, p_end = date.fromisoformat(per["start"]), date.fromisoformat(per["end"])
    big = round(per["median_gap_cfs"], -2)
    outside = s.get("gap_outside_cfs", 0)

    # monthly gaps
    m = figures._monthly_gap(run)
    months, vals = m["month"].to_list(), m["gap"].to_list()
    inside = [p_start <= mo <= p_end for mo in months]
    off_vals = [v for v, i in zip(vals, inside) if i]
    lo, hi = round(min(off_vals), -1), round(max(off_vals), -1)
    xr = (-600, max(hi * 1.1, big * 1.2))
    scale = (f'<div class="scale d">{svg_scale(lo, hi, big, outside, *xr)}</div>'
             f'<div class="scale m">{svg_scale(lo, hi, big, outside, -900, xr[1] * 1.08, W=520)}</div>')
    canal_bands = [(date.fromisoformat(a), date.fromisoformat(b), "var(--heat-soft)") for a, b in CANAL_SEASONS.values()]
    both = lambda fn, *a, **k: f'<div class="plot d">{fn(*a, **k)}</div><div class="plot m">{fn(*a, W=560, **k)}</div>'
    offset_chart = both(svg_bars, months, vals, inside, [])
    canal_chart = both(svg_bars, months, vals, [False] * len(vals), canal_bands, H=230)

    # forecast track record (12 hours ahead, boosted trees vs. "no change")
    f = pl.read_parquet(run / "forecasts_12h.parquet").sort("time")
    y_, g_, b_ = f["y"].to_numpy(), f["gbm"].to_numpy(), f["persistence"].to_numpy()
    avg_m, avg_b = float(np.mean(np.abs(g_ - y_))), float(np.mean(np.abs(b_ - y_)))
    med_m, med_b = float(np.median(np.abs(g_ - y_))), float(np.median(np.abs(b_ - y_)))
    closer = float(np.mean(np.abs(g_ - y_) < np.abs(b_ - y_)))
    cover = float(np.mean((y_ >= f["hybrid_lo"].to_numpy()) & (y_ <= f["hybrid_hi"].to_numpy())))
    w = f.filter(pl.col("time") >= f["time"].max() - np.timedelta64(45, "D")).gather_every(2)
    track = both(svg_track, w["time"].to_list(), w["y"].to_numpy(), w["gbm"].to_numpy(), w["hybrid_lo"].to_numpy(), w["hybrid_hi"].to_numpy())
    recent = f.filter(pl.col("time").dt.hour() == 17).tail(8)
    mx = max(max(abs(r["gbm"] - r["y"]), abs(r["persistence"] - r["y"])) for r in recent.iter_rows(named=True))
    rows = ""
    for r in recent.iter_rows(named=True):
        em, eb = abs(r["gbm"] - r["y"]), abs(r["persistence"] - r["y"])
        win = em < eb
        rows += (f"<tr><td>{r['time']:%a %b %-d, %Y}</td><td>{r['gbm']:,.0f}</td><td>{r['y']:,.0f}</td>"
                 f"<td>{em:,.0f}<span class='missbar{'' if win else ' worse'}' style='width:{60 * em / mx:.0f}px'></span></td>"
                 f"<td>{eb:,.0f}</td><td class='{'win' if win else 'lose'}'>{'Model' if win else 'No change'}</td></tr>")
    grows = "".join(
        f"<tr><td>{e(names[x_['site']])}</td><td>{ids[x_['site']]}</td><td>{q[x_['site']]:,.0f}</td><td>{x_['n']:,}</td><td>{100 * x_['flagged']:.1f}%</td></tr>"
        for x_ in sorted(s["gauges"], key=lambda x_: list(config.SITES).index(x_["site"])))
    repo = f'<a href="{e(REPO_URL)}">Code on GitHub</a>' if REPO_URL else ""
    yrs = f"{s['start'][:4]}–{s['end'][:4]}"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hudson Gauge Check: do the river's flow gauges add up?</title>
<meta name="description" content="A mass-balance audit of three USGS flow gauges at the Hudson–Mohawk confluence, {yrs}, with 12-hour flow forecasts and their track record.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet"><style>{CSS}</style></head><body><div class="wrap">

<header class="mast">
<div class="row"><span class="brand">Hudson Gauge Check</span><span class="kicker wide">Water data</span></div>
<div class="row"><span class="kicker">Mass-balance audit</span><span class="tail"><span class="kicker">{date.today():%B %-d, %Y}</span><span class="wide">{repo}</span></span></div>
</header>

<h1>Do the Hudson River's flow gauges add up?</h1>
<p class="lede">Water is conserved, so the flow past Green Island should equal the upper Hudson plus the Mohawk,
about an hour earlier. This audit checks three U.S. Geological Survey gauges against each other, every 15 minutes from {yrs}.</p>

<article class="call" aria-labelledby="call-h">
<div class="call-head"><span class="pill" id="call-h">Main finding</span>
<p class="when">Gauge <b>Hudson at Green Island</b>, <b>{p_start:%b %Y} – {p_end:%b %Y}</b></p></div>
<div class="call-main">
<div><span class="kicker">Year-long offset</span>
<div class="figline"><div class="fig">+{big:,.0f}</div><div class="unit">cfs</div></div>
<p class="what">Extra flow at Green Island compared with the two rivers upstream</p></div>
<div class="side">
<div><span class="kicker">Monthly range</span><div class="val">+{lo:,.0f} to +{hi:,.0f} <small>cfs</small></div></div>
<div><span class="kicker">Rest of {yrs}</span><div class="val avg">+{outside:,.0f} <small>cfs</small></div></div>
</div></div>
{scale}
<div class="keys"><span><i class="k-pred"></i>Offset-year median</span><span><i class="k-range"></i>Monthly range, offset year</span>
<span><i class="k-avg"></i>Rest of {yrs}</span><span><i class="k-zero"></i>Zero (gauges agree)</span></div>
<div class="notes">
<p><b>Every month of the offset year was above zero.</b> The gap appeared in {p_start:%B %Y} and disappeared in {p_end:%B %Y}; outside that year the gauges agree to within {pct(abs(s.get('gap_outside_pct', 0)))}.</p>
<p><b>The pattern points to a calibration error.</b> A physical cause would follow the seasons or the flow. A step that switches on and off points to one gauge drifting, then being corrected.</p>
</div></article>

<section><span class="kicker">The evidence</span><h2>The gap, month by month</h2>
<p class="sub">Median daily gap between Green Island and the two upstream gauges combined. Months with too few reliable readings are left blank.</p>
<div class="chart"><div class="legend"><span><i class="band" style="background:var(--flame)"></i>Offset year</span><span><i class="band" style="background:var(--faint)"></i>Other months</span><span><i style="border-color:var(--ink)"></i>Zero</span></div>
{offset_chart}</div></section>

<section><span class="kicker">Ruling out the obvious</span><h2>It isn't the Erie Canal</h2>
<p class="sub">The canal diverts part of the Mohawk around its gauge, so it was the first suspect. But the gap is the same with
the canal open ({inv['gap_open_cfs']:,.0f} cfs) as closed ({inv['gap_closed_cfs']:,.0f} cfs), and it moved the expected way at only
{inv['events_consistent']} of {inv['events_total']} openings and closings.</p>
<div class="chart"><div class="legend"><span><i class="band" style="background:var(--heat-soft)"></i>Canal open</span><span><i class="band" style="background:var(--faint)"></i>Monthly gap</span><span><i style="border-color:var(--ink)"></i>Zero</span></div>
{canal_chart}</div></section>

<section><span class="kicker">Model forecast</span><h2>How the forecast has performed</h2>
<p class="sub">A boosted-tree model predicts Green Island's flow 12 hours ahead from upstream flows and rainfall. It was trained only on
earlier data and scored on {len(y_):,} later hours against the simplest baseline: assuming the flow stays the same.</p>
<div class="scores">
<div class="score"><span class="kicker">Average miss</span><div class="v">{avg_m:,.0f} <small>cfs</small></div><div class="l">vs {avg_b:,.0f} cfs for no change, {1 - avg_m / avg_b:.0%} smaller</div></div>
<div class="score"><span class="kicker">Median miss</span><div class="v">{med_m:,.0f} <small>cfs</small></div><div class="l">vs {med_b:,.0f} cfs for no change</div></div>
<div class="score"><span class="kicker">Closer than baseline</span><div class="v">{closer:.0%}</div><div class="l">of hourly forecasts beat no change</div></div>
<div class="score"><span class="kicker">90% range</span><div class="v">{cover:.0%}</div><div class="l">of actual flows fell inside it</div></div>
</div>
<div class="chart"><h3>Forecast against actual flow</h3><p>The last 45 days of the test period.</p>
<div class="legend"><span><i style="border-color:var(--ink)"></i>Actual</span><span><i style="border-color:var(--flame)"></i>Forecast, 12 hours ahead</span><span><i class="band" style="background:var(--flame-soft)"></i>90% range</span></div>
{track}</div>
<h3 class="minor">Recent forecasts</h3><p class="sub">Noon Eastern each day. Flow in cfs.</p>
<div class="tablewrap"><table><thead><tr><th>Day</th><th>Forecast</th><th>Actual</th><th>Model miss</th><th>No-change miss</th><th>Closer</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>

<section><span class="kicker">Method</span><h2>How it works</h2><p class="sub">From raw gauge readings to an audited balance and a forecast.</p>
<ol class="steps">
<li><h3>Collect</h3><p>15-minute readings from three USGS gauges and hourly basin rainfall from Open-Meteo.</p></li>
<li><h3>Clean</h3><p>Flag stuck sensors, spikes, ice and values USGS marks as estimated; {pct(100 * s['flagged'] / s['readings'])} of readings are left out.</p></li>
<li><h3>Balance</h3><p>Compare Green Island with the upstream total, with travel times between gauges fitted from the data.</p></li>
<li><h3>Forecast</h3><p>Predict Green Island 12 hours ahead with boosted trees, tested walk-forward with calibrated ranges.</p></li>
</ol></section>

<section><span class="kicker">Data</span><h2>The three gauges</h2><p class="sub">Median flow over {yrs}.</p>
<div class="tablewrap"><table><thead><tr><th>Gauge</th><th>USGS site</th><th>Median flow (cfs)</th><th>Readings</th><th>Flagged</th></tr></thead><tbody>{grows}</tbody></table></div></section>

<section><span class="kicker">Reference</span><h2>What the numbers mean</h2><p class="sub">The terms used on this page.</p>
<dl class="glossary">
<div><dt>cfs</dt><dd>Cubic feet per second: the volume of water passing a gauge each second. One cubic foot is about 28 litres.</dd></div>
<div><dt>Gap</dt><dd>Flow at Green Island minus the two upstream gauges combined. Zero means the gauges agree.</dd></div>
<div><dt>No change</dt><dd>The baseline forecast: assume the flow 12 hours from now equals the flow now.</dd></div>
<div><dt>90% range</dt><dd>The span the model expects the actual flow to fall inside 90% of the time.</dd></div>
</dl></section>

<footer>Data from the U.S. Geological Survey (gauges {ids['upper_hudson']}, {ids['mohawk']} and {ids['green_island']}) and Open-Meteo.
Built with Python, polars and scikit-learn. {repo}</footer>
</div></body></html>"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    return docs / "index.html"


if __name__ == "__main__":
    path = build(Path.cwd())
    print(f"✅ Website built: {path}")
