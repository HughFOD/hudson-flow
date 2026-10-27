"""Charts and a markdown summary of what the pipeline found."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from . import config
from .balance import col

COLORS = {"upper_hudson": "#2b6cb0", "mohawk": "#c05621", "green_island": "#2f855a"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_hydrograph(hourly, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    for s in config.SITES:
        if col(s) in hourly.columns:
            ax.plot(hourly["time"], hourly[col(s)], lw=0.8, color=COLORS[s],
                    label=config.SITES[s]["name"])
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title("Hourly discharge (bad readings removed)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_balance(bal, path, typical=0.0):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(bal["time"], bal[col(config.OUTFLOW)], lw=0.8, color=COLORS["green_island"],
            label="Measured at Green Island")
    a1.plot(bal["time"], bal["predicted_cfs"], lw=0.8, ls="--", color="k",
            label="Upper Hudson + Mohawk (lagged)")
    a1.set_ylabel("cfs")
    a1.legend(fontsize=8)
    a1.set_title("Flow mass balance")
    a2.plot(bal["time"], bal["residual_pct"], lw=0.6, color="#4a5568")
    a2.axhline(0, color="k", lw=0.5)
    a2.axhline(typical, color="#2b6cb0", lw=0.6, ls="--", label=f"Typical ({typical:+.1f}%)")
    for y in (typical + config.BALANCE_TOL_PCT, typical - config.BALANCE_TOL_PCT):
        a2.axhline(y, color="#e53e3e", lw=0.6, ls=":")
    flagged = bal.filter(pl.col("imbalance"))
    a2.scatter(flagged["time"], flagged["residual_pct"], s=4, color="#e53e3e",
               label="Sustained imbalance")
    # A handful of hours with near-zero outflow give residuals of thousands of
    # percent; zoom to the bulk of the data and say how many points are off-scale.
    r = bal["residual_pct"].drop_nulls()
    if r.len():
        lo, hi = float(r.quantile(0.005)), float(r.quantile(0.995))
        pad = 0.1 * (hi - lo)
        a2.set_ylim(min(lo - pad, typical - 1.5 * config.BALANCE_TOL_PCT),
                    max(hi + pad, typical + 1.5 * config.BALANCE_TOL_PCT))
        n_off = int(((r < a2.get_ylim()[0]) | (r > a2.get_ylim()[1])).sum())
        if n_off:
            a2.text(0.01, 0.03, f"{n_off} extreme hours off-scale", transform=a2.transAxes,
                    fontsize=8, color="#718096")
    a2.set_ylabel("Residual (% of outflow)")
    a2.legend(fontsize=8)
    _save(fig, path)


def plot_conductance(cb, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(cb["time"], cb["cond_measured"], lw=0.8, label="Measured at Green Island")
    ax.plot(cb["time"], cb["cond_predicted"], lw=0.8, ls="--", color="k",
            label="Flow-weighted mix of inflows")
    ax.set_ylabel("Specific conductance (µS/cm)")
    ax.set_title("Component (salt) balance")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_do(do, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for s in do["site"].unique().to_list():
        d = do.filter(pl.col("site") == s)
        ax.plot(d["time"], d["do_sat_pct"], lw=0.8, color=COLORS.get(s), label=s)
    for y in (config.DO_SAT_LOW_PCT, 100, config.DO_SAT_HIGH_PCT):
        ax.axhline(y, color="#e53e3e" if y != 100 else "k", lw=0.6, ls=":")
    ax.set_ylabel("DO (% saturation)")
    ax.set_title("Dissolved oxygen vs. temperature-based saturation")
    ax.legend(fontsize=8)
    _save(fig, path)


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(rows)


def write_report(out: Path, cleaned, hourly, bal, bal_summary, monthly, cb, do):
    out.mkdir(parents=True, exist_ok=True)
    plot_hydrograph(hourly, out / "hydrograph.png")
    plot_balance(bal, out / "flow_balance.png", bal_summary.get("typical_residual_pct", 0.0))
    if cb is not None:
        plot_conductance(cb, out / "conductance_balance.png")
    if do is not None:
        plot_do(do, out / "dissolved_oxygen.png")

    lines = ["# Hudson / Mohawk mass balance report", "",
             f"Period: {hourly['time'].min()} to {hourly['time'].max()}", "",
             "## Data quality flags (15-minute readings)", "",
             _md_table(cleaned["summary"]), "",
             f"Conflicting duplicate timestamps: {cleaned['conflicts'].height}  ",
             f"Gaps longer than 1 hour: {cleaned['gaps'].height}", "",
             "## Flow balance", "", "```", json.dumps(bal_summary, indent=2), "```", "",
             "Median residual by month:", "", _md_table(monthly), "",
             "![flow balance](flow_balance.png)", ""]
    if cb is not None:
        r = cb["cond_residual_pct"].drop_nulls()
        lines += ["## Conductance (salt) balance", "",
                  f"Median residual: {r.median():.2f}%  ",
                  f"Mean absolute residual: {r.abs().mean():.2f}%", "",
                  "![conductance](conductance_balance.png)", ""]
    else:
        lines += ["## Conductance (salt) balance", "",
                  "Skipped: not every site has continuous specific conductance.", ""]
    if do is not None:
        lines += ["## Dissolved oxygen", "",
                  _md_table(do.group_by("site").agg(
                      pl.len().alias("hours"),
                      pl.col("do_sat_pct").median().round(1).alias("median_sat_pct"),
                      pl.col("flag_do_low").sum(), pl.col("flag_do_high").sum())), "",
                  "![dissolved oxygen](dissolved_oxygen.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    cleaned["readings"].write_parquet(out / "readings_flagged.parquet")
    bal.write_parquet(out / "flow_balance.parquet")
    return out / "report.md"
