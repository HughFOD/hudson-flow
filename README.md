# Hudson River Mass Balance: Validating and Forecasting River Data with Physics + ML

**Project website: https://hughfod.github.io/hudson-flow/**

Can I trust a river gauge, and can I predict it? Above Troy, the upper Hudson
and the Mohawk join and flow past a third gauge at Green Island. Water and
dissolved salts are conserved, so the three gauges have to agree. This project
uses that to (1) find bad sensor data, (2) forecast Green Island flow 6–24 hours
ahead, and (3) compare rule-based and machine-learning anomaly detection.

```
Upper Hudson (USGS 01335754) ──┐
                               ├──> Hudson at Green Island (USGS 01358000)
Mohawk at Cohoes (USGS 01357500)┘
```

## 1. Data validation

**Sensor-level cleaning.** Every 15-minute reading gets flags, and nothing is silently dropped:
- USGS qualifiers (ice, equipment malfunction, backwater, maintenance)
- Stuck sensors (flatlines) and one-point spikes, detected on log scale because noise grows with flow
- Gaps, duplicate timestamps with conflicting values, and provisional vs. approved data

**Flow mass balance:** `Q_GreenIsland(t) ≈ Q_Hudson(t − lag₁) + Q_Mohawk(t − lag₂)`
- Travel-time lags are fit **jointly**. Fitting each tributary separately is biased
  when the same weather system hits both basins; the demo shows it picking the wrong lags.
- Sustained departures from the *typical* residual are flagged, not departures from zero,
  because ungauged creeks always add a few percent.

**Component (salt) balance:** `C_out = Σ QᵢCᵢ / Σ Qᵢ`. Conductance is conserved on mixing,
which gives an independent check on the flow data.

**Dissolved oxygen plausibility:** measured DO is compared with saturation at the measured
temperature (Benson & Krause, 1984).

## 2. Forecasting Green Island flow 6, 12 and 24 hours ahead

| Model | What it is |
|---|---|
| persistence | flow in h hours = flow now (the bar to beat) |
| physics | lagged upstream sum × scale factor fit on training data |
| ridge | regularized linear model on 65 features |
| gbm | gradient-boosted trees on the same features |
| hybrid | physics forecast + boosted-tree correction of its error |

Design choices worth explaining in an interview:
- **No leakage.** Features use only data at or before the forecast time. A test scrambles
  the future and checks that past features don't change.
- **Walk-forward evaluation** with an embargo equal to the horizon. scikit-learn's default
  early stopping uses a *random* validation split, which leaks future data on time series,
  so it's turned off.
- **Log-ratio targets** (`log(y / reference)`): a storm doubling the flow looks the same at
  any flow level.
- **Metrics hydrologists use:** Nash–Sutcliffe efficiency, skill vs. persistence, and error
  split into storm and baseflow, since average error hides storm performance.
- **Ablations:** retraining without rainfall, and without upstream gauges, shows what each
  information source is worth.
- **Calibrated uncertainty:** quantile models give a 90% interval, and conformal calibration
  (Romano et al., 2019) fixes their overconfidence.
- **Permutation importance** shows what the hybrid model relies on.

## 3. Anomaly detection: rules vs. Isolation Forest

The Isolation Forest scores how unusual each hour is from **storm-invariant** signals:
balance residual, its drift from the previous two weeks, and each gauge's noise relative
to the downstream gauge. Raw rates of change would make every storm look anomalous.
Flags must persist for 3+ hours. Its output is compared hour by hour with the rules
(both / rules only / model only), and model-only episodes are listed for investigation.

## Demo results (synthetic data, not the real river)

`python run.py demo` runs everything on two years of simulated data with planted faults,
including a slow calibration drift that has no qualifier and no stuck values:

- Sensor-level rules catch the ice period, flatline, spike, gap and duplicate, but **none** of the drift.
- The mass balance and the Isolation Forest both catch the drift.
- Joint lag fitting recovers the true travel times (3 h, 2 h); pairwise fitting gets them wrong.
- At 6 h ahead, the boosted models beat persistence by ~82% and the physics baseline by ~65%.
- At 24 h ahead, **ridge beats the tree models**. The simulated river is a linear system, and
  trees can't extrapolate. Don't assume this holds for the real river; test it.
- Conformal calibration brings 90% interval coverage from 64–80% up to ~90%.

## Known physical complications on the real river

- **Barge Canal diversion:** the Mohawk gauge sits downstream of where water is diverted
  into the canal at Crescent Dam. The canal runs only part of the year, so check the
  monthly residual table for a seasonal pattern.
- **Tides:** below the Troy dam the Hudson is tidal, which is why the balance point is above it.
- **Ice:** upstate winters produce ice-affected readings, which are flagged and excluded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest              # tests, ~1 minute
python run.py demo            # full pipeline on synthetic data, ~3 minutes, no internet
python run.py demo --no-ml    # validation only, a few seconds
```

## Real data

```bash
python run.py inventory                                    # what each gauge measures
python run.py analyze --start 2022-01-01 --end 2025-12-31  # download, validate, forecast
```

Use at least 3 years of data so the models see each season more than once. River data
comes from the USGS Water Data API through `dataretrieval.waterdata`; the legacy NWIS
services are being retired in early 2027, so this project doesn't use them. Rainfall comes
from the free Open-Meteo historical archive. Downloads are cached in `data/raw/`, and
reports go to `output/`.

## Findings on the real river

*(Write this once you've run it. Specific numbers make it convincing.)*

- What fraction of readings got flagged, and for what?
- Does the balance close? Is the residual seasonal, and does that match the canal season?
- Do the flow and salt balances agree on which periods look wrong?
- Which model wins at each horizon, and does rainfall help on the real river?
- What did the anomaly model find that the rules missed, and was it real?

## Project layout

```
hudson/config.py       gauges, parameters, every threshold
hudson/fetch.py        USGS download with per-year parquet cache
hudson/weather.py      basin-average rainfall from Open-Meteo
hudson/clean.py        flags, dedup, gaps, hourly aggregation
hudson/balance.py      joint lag fitting and flow mass balance
hudson/chemistry.py    conductance mixing balance and DO saturation
hudson/features.py     leakage-safe forecasting features
hudson/forecast.py     models, walk-forward evaluation, conformal intervals
hudson/anomaly.py      Isolation Forest vs. rule-based flags
hudson/report.py       validation charts and report
hudson/ml_report.py    forecasting and anomaly charts and report
hudson/synthetic.py    simulated river with known faults, for tests and the demo
run.py                 command-line entry point
tests/                 checks every validator, the leakage guarantees, and the models
```
