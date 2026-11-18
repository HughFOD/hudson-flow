"""Figures for the project website, drawn in one restrained style.

Each function reads the pipeline's output files and writes a PNG.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .investigate import CANAL_SEASONS

INK, MUTED, RULE, ACCENT, ACCENT_LIGHT, SHADE = "#1a1a1a", "#6b6b6b", "#d9d9d9", "#1d4e89", "#c9d8ea", "#efefef"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
OUT = f"{config.OUTFLOW}__{config.FLOW}"


def _daily(run: Path) -> pl.DataFrame:
    b = pl.read_parquet(run / "flow_balance.parquet").filter(~pl.col("incomplete"))
    return (b.group_by(pl.col("time").dt.date().alias("d"))
             .agg(pl.col(OUT).mean().alias("measured"), pl.col("predicted_cfs").mean().alias("upstream"), pl.len().alias("h"))
             .filter(pl.col("h") >= 20).sort("d"))


def _thousands(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if abs(v) >= 1000 else f"{v:,.0f}"))


def _with_breaks(d: pl.DataFrame) -> pl.DataFrame:
    """Insert empty rows where days are missing, so lines break instead of bridging gaps."""
    full = pl.DataFrame({"d": pl.date_range(d["d"].min(), d["d"].max(), "1d", eager=True)})
    return full.join(d, on="d", how="left")


def _years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _offset_band(ax, per):
    if per:
        ax.axvspan(date.fromisoformat(per["start"]), date.fromisoformat(per["end"]), color=SHADE, zorder=0, lw=0)


def balance(run: Path, per: dict | None, path: Path):
    d = _with_breaks(_daily(run))
    x = d["d"].to_list()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.12})
    _offset_band(a1, per)
    a1.plot(x, d["upstream"], color=MUTED, lw=0.9, label="Upper Hudson + Mohawk (upstream)")
    a1.plot(x, d["measured"], color=ACCENT, lw=0.9, label="Green Island (downstream)")
    a1.set_ylabel("flow (cfs)")
    _thousands(a1)
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    gap = (d["measured"] - d["upstream"]).to_numpy().astype(float)
    _offset_band(a2, per)
    a2.axhline(0, color=INK, lw=0.7)
    a2.plot(x, gap, color=INK, lw=0.7)
    lo, hi = np.nanpercentile(gap, [0.5, 99.5])
    a2.set_ylim(min(lo, -500) * 1.1, max(hi, 500) * 1.1)
    a2.set_ylabel("gap (cfs)")
    _thousands(a2)
    _years(a2)
    fig.savefig(path)
    plt.close(fig)


def monthly(run: Path, per: dict | None, path: Path):
    d = _daily(run).with_columns((pl.col("measured") - pl.col("upstream")).alias("gap"))
    m = (d.group_by(pl.col("d").dt.truncate("1mo").alias("month")).agg(pl.col("gap").median()).sort("month"))
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    for y, (a, b) in CANAL_SEASONS.items():
        ax.axvspan(date.fromisoformat(a), date.fromisoformat(b), color=ACCENT_LIGHT, alpha=0.45, lw=0, zorder=0)
    months = m["month"].to_list()
    inside = [bool(per) and date.fromisoformat(per["start"]) <= mo <= date.fromisoformat(per["end"]) for mo in months]
    ax.bar(months, m["gap"], width=24, color=[INK if i else "#9a9a9a" for i in inside], lw=0)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("median daily gap (cfs)")
    _thousands(ax)
    _years(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def storm(run: Path, path: Path, horizon: int = 12):
    f = pl.read_parquet(run / f"forecasts_{horizon}h.parquet").sort("time")
    peak = f.filter(pl.col("y") == f["y"].max())["time"][0]
    w = f.filter(pl.col("time").is_between(peak - np.timedelta64(5, "D"), peak + np.timedelta64(6, "D")))
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    t = w["time"].to_list()
    ax.fill_between(t, w["hybrid_lo"], w["hybrid_hi"], color=ACCENT_LIGHT, lw=0, label="90% range")
    ax.plot(t, w["y"], color=INK, lw=1.3, label="measured")
    ax.plot(t, w["hybrid"], color=ACCENT, lw=1.1, label=f"forecast made {horizon} hours earlier")
    ax.set_ylim(0, float(w["y"].max()) * 1.45)
    ax.set_ylabel("Green Island flow (cfs)")
    _thousands(ax)
    ax.legend(loc="upper left", ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    fig.savefig(path)
    plt.close(fig)


# ---------------- dark versions, matched to the website's cards ----------------
CARD, D_INK, D_MUTED, D_GRID, BLUE, BLUE_LIGHT, GREY, ZERO = "#121721", "#e8ebf0", "#8b94a3", "#1d2430", "#5b9bff", "#22406e", "#4a5262", "#c9ced6"


def _dark(ax, fig):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(D_GRID)
    ax.tick_params(colors=D_MUTED, labelsize=9)
    ax.yaxis.label.set_color(D_MUTED)
    ax.grid(color=D_GRID, lw=0.7)
    ax.grid(axis="x", visible=False)


def _monthly_gap(run: Path) -> pl.DataFrame:
    d = _daily(run).with_columns((pl.col("measured") - pl.col("upstream")).alias("gap"))
    return d.group_by(pl.col("d").dt.truncate("1mo").alias("month")).agg(pl.col("gap").median()).sort("month")


def hero_gap(run: Path, per: dict, outside_cfs: float, path: Path):
    m = _monthly_gap(run)
    months = m["month"].to_list()
    a, b = date.fromisoformat(per["start"]), date.fromisoformat(per["end"])
    inside = [a <= mo <= b for mo in months]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _dark(ax, fig)
    ax.bar(months, m["gap"], width=24, color=[BLUE if i else GREY for i in inside], lw=0, zorder=2)
    ax.axhline(0, color=ZERO, lw=1.3, zorder=3)
    big = per["median_gap_cfs"]
    ax.hlines(big, a, b, color=BLUE, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(b, big, f"  +{round(big, -2):,.0f} cfs", color=BLUE, fontsize=10.5, fontweight="bold", va="center")
    mid = a + (b - a) / 2
    ax.text(mid, max(m["gap"].max(), big) * 1.13, f"{a:%b %Y} – {b:%b %Y}", color=D_MUTED, fontsize=8.5, ha="center")
    ax.set_ylim(min(m["gap"].min() * 1.15, -800), max(m["gap"].max(), big) * 1.25)
    ax.set_ylabel("monthly gap (cfs)")
    _thousands(ax)
    _years(ax)
    fig.savefig(path, facecolor=CARD)
    plt.close(fig)


def canal_dark(run: Path, path: Path):
    m = _monthly_gap(run)
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    _dark(ax, fig)
    for y, (a, b) in CANAL_SEASONS.items():
        ax.axvspan(date.fromisoformat(a), date.fromisoformat(b), color="#1b3b38", lw=0, zorder=0)
    ax.bar(m["month"].to_list(), m["gap"], width=24, color=GREY, lw=0, zorder=2)
    ax.axhline(0, color=ZERO, lw=1.3, zorder=3)
    ax.set_ylabel("monthly gap (cfs)")
    _thousands(ax)
    _years(ax)
    fig.savefig(path, facecolor=CARD)
    plt.close(fig)


def track_dark(run: Path, path: Path, horizon: int = 12, days: int = 45):
    f = pl.read_parquet(run / f"forecasts_{horizon}h.parquet").sort("time")
    end = f["time"].max()
    w = f.filter(pl.col("time") >= end - np.timedelta64(days, "D"))
    t = w["time"].to_list()
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    _dark(ax, fig)
    ax.fill_between(t, w["hybrid_lo"], w["hybrid_hi"], color=BLUE_LIGHT, lw=0, alpha=0.8)
    ax.plot(t, w["y"], color=D_INK, lw=1.4)
    ax.plot(t, w["gbm"], color=BLUE, lw=1.2)
    ax.set_ylim(0, float(np.nanmax(w["y"].to_numpy())) * 1.35)
    ax.set_ylabel("Green Island flow (cfs)")
    _thousands(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    fig.savefig(path, facecolor=CARD)
    plt.close(fig)
