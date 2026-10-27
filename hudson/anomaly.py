"""Unsupervised anomaly detection, compared against the rule-based flags.

The rules in clean.py catch problems you can describe in advance: stuck
sensors, spikes, USGS qualifiers. They can't catch a sensor that slowly
drifts out of calibration while still reporting plausible-looking numbers.

An Isolation Forest learns what "normal" looks like across several
physically meaningful signals at once (balance residual, how fast each gauge
is changing, how noisy each gauge is) and scores how unusual each hour is.
Comparing its output with the rules shows:
  - hours both catch           -> confirmed problems
  - rules only                 -> problems the model considers normal (e.g. short spikes)
  - model only                 -> candidates for problems the rules missed (inspect these!)

It's unsupervised, so there's no ground truth on real data: the model-only
hours are leads to investigate, not proven errors.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from . import config
from .balance import col


def build_signals(hourly_all: pl.DataFrame, lags: dict) -> pl.DataFrame:
    """Hourly signals from RAW data (flagged readings included).

    Every signal is designed to be STORM-INVARIANT. A storm makes all gauges
    change fast at once, which is normal; a broken gauge behaves differently
    from its neighbours. Raw rates of change would make every storm look
    anomalous and drown out the real problems.
    """
    out = pl.col(col(config.OUTFLOW))
    pred = pl.sum_horizontal([pl.col(col(s)).shift(lags[s]["lag_hours"]) for s in config.INFLOWS])
    r = (100 * (out - pred) / out)

    def noise(c):  # short-term variability of hourly log-flow changes
        return pl.col(c).log().diff().rolling_std(6, min_samples=4)

    out_noise = noise(col(config.OUTFLOW))
    exprs = [
        pl.col("time"),
        r.alias("balance_resid_pct"),
        r.diff().alias("balance_resid_change"),
        # how far the residual is from its normal over the PREVIOUS two weeks;
        # a slow calibration drift accumulates here
        (r - r.shift(24).rolling_median(24 * 14, min_samples=48)).alias("balance_resid_vs_2wk"),
    ]
    for s in config.INFLOWS:
        # this gauge's variability relative to the downstream gauge's:
        # ~0 normally (storms hit both), very negative if stuck, positive if erratic
        exprs.append((noise(col(s)) + 1e-4).log().sub((out_noise + 1e-4).log())
                     .alias(f"{s}_noise_vs_outflow"))
    if col(config.OUTFLOW, "spec_cond_uScm") in hourly_all.columns:
        exprs.append(pl.col(col(config.OUTFLOW, "spec_cond_uScm")).log().diff()
                     .alias("outflow_dlog_cond"))
    return hourly_all.select(exprs)


MIN_COVERAGE = 0.5  # a signal must exist in at least half of all hours to be used


def detect(signals: pl.DataFrame) -> pl.DataFrame:
    # Drop sparse signals first. The model needs every signal in every hour it
    # scores, so one sparse signal would throw away almost all the hours.
    # Real example: Green Island measured conductance for only six weeks in
    # 2024, during its flow-data gap, which left ZERO complete hours.
    cand = [c for c in signals.columns if c != "time"]
    cover = signals.select([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cand]).mean()
    names = [c for c in cand if cover[c][0] >= MIN_COVERAGE]
    dropped = [c for c in cand if c not in names]
    if dropped:
        print(f"  anomaly detection: skipping sparse signals {dropped}")
    complete = signals.drop_nulls(names).filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in names]))
    if complete.height < 100:
        print(f"  anomaly detection: only {complete.height} complete hours, skipping")
        return signals.select("time").with_columns(
            pl.lit(None, pl.Float64).alias("anomaly_score"), pl.lit(False).alias("ml_raw"),
            pl.lit(None, pl.String).alias("main_driver"), pl.lit(False).alias("ml_anomaly"))
    X = complete.select(names).to_numpy()
    # standardize robustly so no single signal dominates
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    Xs = (X - med) / np.where(iqr > 0, iqr, 1.0)

    forest = IsolationForest(n_estimators=300, contamination=config.ANOMALY_CONTAMINATION,
                             random_state=0).fit(Xs)
    score = -forest.score_samples(Xs)  # higher = more anomalous
    # which signal is most extreme in each hour: a first guess at "why"
    driver = np.array(names)[np.argmax(np.abs(Xs), axis=1)]
    res = complete.select("time").with_columns(
        pl.Series("anomaly_score", score),
        pl.Series("ml_raw", forest.predict(Xs) == -1),
        pl.Series("main_driver", driver),
    )
    out = signals.select("time").join(res, on="time", how="left").sort("time")
    # Isolation Forest always flags a fixed fraction of hours, so isolated
    # flags are mostly noise. Keep only runs of >= MIN_RUN consecutive hours.
    f = pl.col("ml_raw").fill_null(False)
    run_end = f.cast(pl.Int32).rolling_sum(MIN_RUN) == MIN_RUN
    in_run = pl.any_horizontal([run_end.shift(-k) for k in range(MIN_RUN)]).fill_null(False)
    return out.with_columns(in_run.alias("ml_anomaly"))


MIN_RUN = 3


def rule_flags_hourly(readings: pl.DataFrame, balance: pl.DataFrame) -> pl.DataFrame:
    sensor = (
        readings.filter(pl.col("param") == config.FLOW)
        .group_by(pl.col("time").dt.truncate("1h"))
        .agg(pl.col("bad").any().alias("rule_sensor"))
    )
    return (balance.select("time", pl.col("imbalance").alias("rule_balance"))
            .join(sensor, on="time", how="left")
            .with_columns(pl.col("rule_sensor").fill_null(False)))


def compare(detected: pl.DataFrame, rules: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (hour-level comparison, summary counts)."""
    df = detected.join(rules, on="time", how="left").with_columns(
        pl.col("ml_anomaly").fill_null(False),
        (pl.col("rule_sensor") | pl.col("rule_balance")).fill_null(False).alias("rule_any"),
    ).with_columns(
        pl.when(pl.col("ml_anomaly") & pl.col("rule_any")).then(pl.lit("both"))
        .when(pl.col("rule_any")).then(pl.lit("rules only"))
        .when(pl.col("ml_anomaly")).then(pl.lit("model only"))
        .otherwise(pl.lit("neither")).alias("agreement")
    )
    summary = df.group_by("agreement").agg(pl.len().alias("hours")).sort("hours", descending=True)
    return df, summary


def episodes(compared: pl.DataFrame, which: str = "model only", min_hours: int = 3) -> pl.DataFrame:
    """Group consecutive flagged hours into episodes, longest first."""
    d = compared.sort("time").with_columns((pl.col("agreement") == which).alias("_f"))
    d = d.with_columns((pl.col("_f") != pl.col("_f").shift(1)).cum_sum().alias("_grp"))
    return (
        d.filter(pl.col("_f")).group_by("_grp").agg(
            pl.col("time").min().alias("start"), pl.col("time").max().alias("end"),
            pl.len().alias("hours"), pl.col("anomaly_score").max().round(3).alias("max_score"),
            pl.col("main_driver").mode().first().alias("main_driver"),
        ).filter(pl.col("hours") >= min_hours).drop("_grp").sort("hours", descending=True)
    )
