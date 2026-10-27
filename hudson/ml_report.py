"""Charts and a markdown report for the forecasting and anomaly experiments."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .balance import col
from .report import _md_table, _save

MODEL_STYLE = {
    "persistence": ("#a0aec0", "-"), "physics": ("#2b6cb0", "-"), "ridge": ("#805ad5", "-"),
    "gbm": ("#dd6b20", "-"), "hybrid": ("#2f855a", "-"),
    "gbm_no_rain": ("#dd6b20", ":"), "gbm_no_upstream": ("#dd6b20", "--"),
}


def plot_skill(results, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = pl.concat([r["table"] for r in results])
    for m, (c, ls) in MODEL_STYLE.items():
        d = table.filter(pl.col("model") == m).sort("horizon_h")
        ax.plot(d["horizon_h"], d["skill_vs_persistence"], marker="o", color=c, ls=ls, label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Skill vs. persistence (1 − MAE/MAE_persistence)")
    ax.set_title("Forecast skill by horizon (walk-forward test periods)")
    ax.set_xticks([r["horizon"] for r in results])
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_storm(result, path):
    """The largest storm in the test periods: forecasts vs. what happened."""
    p = result["preds"].sort("time")
    peak_t = p.filter(pl.col("y") == p["y"].max())["time"][0]
    w = p.filter(pl.col("time").is_between(peak_t - timedelta(days=4), peak_t + timedelta(days=4)))
    # x-axis is the time the forecast is FOR (t + h), so lines align with reality
    tt = w.select(pl.col("time") + pl.duration(hours=result["horizon"]))["time"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(tt, w["hybrid_lo"], w["hybrid_hi"], color="#2f855a", alpha=0.15,
                    label="Hybrid 90% interval (conformal)")
    ax.plot(tt, w["y"], color="k", lw=1.6, label="Observed")
    for m in ("persistence", "physics", "hybrid"):
        c, ls = MODEL_STYLE[m]
        ax.plot(tt, w[m], color=c, ls=ls, lw=1.1, label=m)
    ax.set_ylabel("Green Island discharge (cfs)")
    ax.set_title(f"Largest test-period storm, {result['horizon']} h ahead forecasts")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_importance(result, path, top=15):
    imp = result["importance"].head(top).reverse()
    fig, ax = plt.subplots(figsize=(8, 0.35 * top + 1.2))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#2f855a")
    ax.set_xlabel("Increase in error when shuffled (log-ratio MAE)")
    ax.set_title(f"Permutation importance, hybrid model, {result['horizon']} h ahead")
    _save(fig, path)


def plot_anomalies(compared, bal, path, typical=0.0):
    d = bal.select("time", "residual_pct").join(compared.select("time", "agreement"),
                                                on="time", how="left")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["time"], d["residual_pct"], lw=0.5, color="#a0aec0")
    for label, c in [("rules only", "#3182ce"), ("model only", "#d69e2e"), ("both", "#e53e3e")]:
        s = d.filter(pl.col("agreement") == label)
        ax.scatter(s["time"], s["residual_pct"], s=6, color=c, label=label, zorder=3)
    ax.axhline(typical, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Flow balance residual (%)")
    ax.set_title("Anomalies: rule-based checks vs. Isolation Forest")
    ax.legend(fontsize=8)
    _save(fig, path)


def write(out: Path, results, compared, agree_summary, model_episodes, bal, bal_summary,
          has_rain: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    plot_skill(results, out / "skill_by_horizon.png")
    mid = results[len(results) // 2]
    plot_storm(mid, out / "storm_forecast.png")
    plot_importance(mid, out / "feature_importance.png")
    typical = bal_summary.get("typical_residual_pct", 0.0)
    plot_anomalies(compared, bal, out / "anomalies.png", typical)

    table = pl.concat([r["table"] for r in results]).select(
        "horizon_h", "model", pl.col("MAE_cfs").round(0), pl.col("NSE").round(3),
        pl.col("skill_vs_persistence").round(3), pl.col("MAE_storm").round(0),
        pl.col("MAE_baseflow").round(0), pl.col("bias_cfs").round(0))
    intervals = pl.DataFrame([{"horizon_h": r["horizon"], **r["interval"]} for r in results])

    lines = [
        "# Forecasting and anomaly detection", "",
        f"Target: Green Island discharge, {', '.join(str(r['horizon']) for r in results)} hours ahead.  ",
        f"Rows: {results[0]['n_rows']:,} hours, {results[0]['n_features']} features, "
        f"rainfall {'included' if has_rain else 'NOT available'}.  ",
        f"Evaluation: walk-forward, {config.N_FOLDS} test blocks covering the last "
        f"{int(100 * (1 - config.INITIAL_TRAIN_FRAC))}% of the record, embargo = horizon.  ",
        f"Storm = flow above the {int(100 * config.STORM_QUANTILE)}th percentile of the first training set "
        f"({results[0]['storm_thresh']:,.0f} cfs).", "",
        "## Model comparison", "",
        "Skill > 0 means better than persistence. NSE is Nash–Sutcliffe efficiency "
        "(1 = perfect, 0 = no better than the mean).", "",
        _md_table(table), "",
        "![skill](skill_by_horizon.png)", "",
        "## Ablations: what information matters?", "",
        "`gbm_no_rain` drops rainfall features; `gbm_no_upstream` drops upstream gauges and rainfall, "
        "leaving only Green Island's own history. The gap to `gbm` is what that information is worth.", "",
        "## Prediction intervals", "",
        f"Nominal coverage {intervals['nominal'][0]:.0%}. Raw quantile models vs. after conformal calibration:", "",
        _md_table(intervals), "",
        "![storm](storm_forecast.png)", "",
        "## What the hybrid model relies on", "",
        "![importance](feature_importance.png)", "",
        "## Anomaly detection: rules vs. Isolation Forest", "",
        _md_table(agree_summary), "",
        "Longest episodes flagged by the model but not by any rule (inspect these first):", "",
        _md_table(model_episodes.head(10)) if model_episodes.height else "_None._", "",
        "![anomalies](anomalies.png)", "",
    ]
    path = out / "ml_report.md"
    path.write_text("\n".join(lines))
    for r in results:
        r["preds"].write_parquet(out / f"forecasts_{r['horizon']}h.parquet")
    return path
