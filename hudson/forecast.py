"""Forecast Green Island flow 6-24 hours ahead, evaluated honestly.

Models, from simplest to most complex. Each must beat the ones above it to
justify itself:

  persistence  flow in h hours = flow now
  physics      mass balance: lagged upstream flows x a scale factor fit on training data
  ridge        regularized linear model on all features, predicting the change from now
  gbm          gradient-boosted trees on all features, predicting the change from now
  hybrid       physics forecast + gradient-boosted correction of its error,
               with quantile models for a 90% prediction interval

Evaluation is walk-forward: train on everything before a test block, predict
the block, move on. Training rows whose target falls inside the test block are
dropped (an "embargo" of h hours), otherwise the last training targets would
be test-period values.

Note: scikit-learn's HistGradientBoosting turns on early stopping with a
RANDOM validation split for large datasets. On time series that leaks the
future into training, so it's switched off explicitly here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features

MODELS = ["persistence", "physics", "ridge", "gbm", "hybrid"]
ABLATIONS = {"gbm_no_rain": ("rain",), "gbm_no_upstream": ("upstream", "rain")}


def _gbm(loss="squared_error", quantile=None):
    kw = dict(loss=loss, max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=False, random_state=0)
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 4, 20)))


@dataclass
class FoldResult:
    preds: pl.DataFrame                      # time, y, persist, one column per model (+ lo/hi)
    hybrid_model: object = None
    X_test: np.ndarray | None = None
    target_test: np.ndarray | None = None
    names: list[str] = field(default_factory=list)


def _log(a):
    return np.log(np.clip(a, 1.0, None))


def _fit_predict(train: pl.DataFrame, test: pl.DataFrame, names: list[str],
                 horizon: int) -> FoldResult:
    """Fit every model on `train`, predict `test`.

    ML targets are LOG RATIOS: log(y / reference), where the reference is
    persistence or the physics forecast. A storm doubling the flow looks the
    same at 3,000 cfs as at 30,000 cfs, so the model generalizes across flow
    levels instead of having to learn each one separately.
    """
    Xtr, Xte = train.select(names).to_numpy(), test.select(names).to_numpy()
    ytr = train["y"].to_numpy()
    p_tr, p_te = train["persist"].to_numpy(), test["persist"].to_numpy()
    out = {"time": test["time"], "y": test["y"], "persist": p_te}

    out["persistence"] = p_te

    ok = train["phys_raw"].is_not_null().to_numpy() & (train["phys_raw"].to_numpy() > 0)
    scale = float(np.median(ytr[ok] / train["phys_raw"].to_numpy()[ok])) if ok.any() else 1.0
    phys_tr = scale * train["phys_raw"].to_numpy()
    phys_te = scale * test["phys_raw"].to_numpy()
    # fall back to persistence where an upstream reading is missing
    phys_tr = np.where(np.isnan(phys_tr), p_tr, phys_tr)
    phys_te = np.where(np.isnan(phys_te), p_te, phys_te)
    out["physics"] = phys_te

    t_persist = _log(ytr) - _log(p_tr)
    # Never predict a change bigger than any seen in training. On the real
    # river one unusual hour sent the linear model to exp(huge) and an
    # average error of 24,000 cfs; trees can't extrapolate, linear models can,
    # badly. Clipping to the training range is the standard guard.
    clip_p = (t_persist.min(), t_persist.max())
    ridge = _ridge().fit(Xtr, t_persist)
    out["ridge"] = p_te * np.exp(np.clip(ridge.predict(Xte), *clip_p))

    gbm = _gbm().fit(Xtr, t_persist)
    out["gbm"] = p_te * np.exp(np.clip(gbm.predict(Xte), *clip_p))

    t_phys = _log(ytr) - _log(phys_tr)
    clip_h = (t_phys.min(), t_phys.max())
    hybrid = _gbm().fit(Xtr, t_phys)
    out["hybrid"] = phys_te * np.exp(np.clip(hybrid.predict(Xte), *clip_h))
    lo, hi, lo_raw, hi_raw = _conformal_interval(train, Xtr, t_phys, Xte, horizon)
    out["hybrid_lo"], out["hybrid_hi"] = phys_te * np.exp(lo), phys_te * np.exp(hi)
    out["hybrid_lo_raw"], out["hybrid_hi_raw"] = phys_te * np.exp(lo_raw), phys_te * np.exp(hi_raw)

    target_te = _log(test["y"].to_numpy()) - _log(phys_te)
    return FoldResult(pl.DataFrame(out), hybrid, Xte, target_te, names)


def _conformal_interval(train, Xtr, target, Xte, horizon, calib_frac=0.2):
    """Conformalized quantile regression (Romano et al., 2019).

    Raw quantile models are usually overconfident on new data. So: fit them
    on the earlier 80% of the training period, measure how far real outcomes
    fall outside their interval on the most recent 20%, and widen the interval
    by that amount. The calibration slice comes AFTER the fitting slice (with
    an embargo), mimicking how the model will actually be used.
    """
    lo_q, hi_q = config.INTERVAL
    t = train["time"].to_numpy()
    cal_start = t[int(len(t) * (1 - calib_frac))]
    fit_m = t <= cal_start - np.timedelta64(horizon, "h")
    cal_m = t >= cal_start
    q_lo = _gbm("quantile", lo_q).fit(Xtr[fit_m], target[fit_m])
    q_hi = _gbm("quantile", hi_q).fit(Xtr[fit_m], target[fit_m])

    e = np.maximum(q_lo.predict(Xtr[cal_m]) - target[cal_m],
                   target[cal_m] - q_hi.predict(Xtr[cal_m]))
    n = len(e)
    level = min(1.0, (hi_q - lo_q) * (1 + 1 / n))
    pad = float(np.quantile(e, level))
    lo_raw, hi_raw = q_lo.predict(Xte), q_hi.predict(Xte)
    return lo_raw - pad, hi_raw + pad, lo_raw, hi_raw


def _ablation_predict(train, test, names):
    Xtr, Xte = train.select(names).to_numpy(), test.select(names).to_numpy()
    p_tr = train["persist"].to_numpy()
    m = _gbm().fit(Xtr, _log(train["y"].to_numpy()) - _log(p_tr))
    return test["persist"].to_numpy() * np.exp(m.predict(Xte))


def walk_forward_splits(times: pl.Series, horizon: int):
    """Yield (train_mask, test_mask) with an embargo of `horizon` hours."""
    n = len(times)
    start = int(n * config.INITIAL_TRAIN_FRAC)
    block = (n - start) // config.N_FOLDS
    t = times.to_numpy()
    for k in range(config.N_FOLDS):
        a = start + k * block
        b = n if k == config.N_FOLDS - 1 else a + block
        test = np.zeros(n, bool)
        test[a:b] = True
        cutoff = t[a] - np.timedelta64(horizon, "h")
        train = t <= cutoff
        yield train, test


def metrics(y, pred, persist, storm_thresh) -> dict:
    y, pred, persist = map(np.asarray, (y, pred, persist))
    err = pred - y
    mae = np.mean(np.abs(err))
    storm = y > storm_thresh
    return {
        "MAE_cfs": mae,
        "RMSE_cfs": np.sqrt(np.mean(err ** 2)),
        "bias_cfs": np.mean(err),
        # Nash-Sutcliffe efficiency: hydrology's standard skill score. 1 = perfect,
        # 0 = no better than predicting the mean, negative = worse than the mean.
        "NSE": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
        "skill_vs_persistence": 1 - mae / np.mean(np.abs(persist - y)),
        "MAE_storm": np.mean(np.abs(err[storm])) if storm.any() else np.nan,
        "MAE_baseflow": np.mean(np.abs(err[~storm])) if (~storm).any() else np.nan,
    }


def run_horizon(hourly: pl.DataFrame, lags: dict, horizon: int) -> dict:
    df = features.build(hourly, lags, horizon)
    df = df.filter(pl.col("y").is_not_null() & pl.col("persist").is_not_null())
    names = features.feature_names(df)
    ablation_names = {k: features.feature_names(df, v) for k, v in ABLATIONS.items()}

    folds, ablation_preds = [], {k: [] for k in ABLATIONS}
    storm_thresh = None
    for train_m, test_m in walk_forward_splits(df["time"], horizon):
        train, test = df.filter(pl.Series(train_m)), df.filter(pl.Series(test_m))
        if storm_thresh is None:  # from the FIRST training set only
            storm_thresh = float(train["y"].quantile(config.STORM_QUANTILE))
        folds.append(_fit_predict(train, test, names, horizon))
        for k, nm in ablation_names.items():
            ablation_preds[k].append(_ablation_predict(train, test, nm))

    preds = pl.concat([f.preds for f in folds])
    for k in ABLATIONS:
        preds = preds.with_columns(pl.Series(k, np.concatenate(ablation_preds[k])))

    rows = []
    for m in MODELS + list(ABLATIONS):
        r = metrics(preds["y"], preds[m], preds["persist"], storm_thresh)
        rows.append({"horizon_h": horizon, "model": m, **r})
    table = pl.DataFrame(rows).with_columns(pl.selectors.float().round(3))

    y = preds["y"].to_numpy()
    lo, hi = preds["hybrid_lo"].to_numpy(), preds["hybrid_hi"].to_numpy()
    lo_r, hi_r = preds["hybrid_lo_raw"].to_numpy(), preds["hybrid_hi_raw"].to_numpy()
    storm = y > storm_thresh

    def _cov(a, b, m=slice(None)):
        return round(float(np.mean(((y >= a) & (y <= b))[m])), 3)

    interval = {
        "nominal": round(config.INTERVAL[1] - config.INTERVAL[0], 3),
        "coverage_raw_quantile": _cov(lo_r, hi_r),
        "coverage_conformal": _cov(lo, hi),
        "coverage_conformal_storm": _cov(lo, hi, storm),
        "mean_width_pct_conformal": round(float(np.mean((hi - lo) / y) * 100), 1),
    }

    last = folds[-1]
    imp = permutation_importance(
        last.hybrid_model, last.X_test, last.target_test, n_repeats=3,
        random_state=0, scoring="neg_mean_absolute_error",
        max_samples=min(2000, len(last.target_test)),
    )
    importance = pl.DataFrame({
        "feature": last.names, "importance": imp.importances_mean, "std": imp.importances_std,
    }).sort("importance", descending=True)

    return {"horizon": horizon, "table": table, "preds": preds, "interval": interval,
            "importance": importance, "storm_thresh": storm_thresh,
            "n_features": len(names), "n_rows": df.height}


def run(hourly: pl.DataFrame, lags: dict, horizons=config.HORIZONS) -> list[dict]:
    results = []
    for h in horizons:
        print(f"  forecasting {h} h ahead...")
        results.append(run_horizon(hourly, lags, h))
    return results
