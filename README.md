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
  into the canal at Crescent Dam. Tested against the real season dates, it is too
  small to explain the gap (see Findings).
- **Tides:** below the Troy dam the Hudson is tidal, which is why the balance point is above it.
- **Ice:** upstate winters produce ice-affected readings, which are flagged and excluded.

## Running it in PyCharm (no terminal needed)

Open each file and click the green ▶ next to `if __name__ == "__main__":`

| File | What it does |
|---|---|
| `step1_setup.py` | creates `.venv` and installs everything (once) |
| `step2_run_demo.py` | full pipeline on synthetic data with planted faults |
| `step3_run_real_data.py` | downloads 2022–2025 USGS data and runs everything |
| `step4_run_tests.py` | automated checks |
| `step5_export_spreadsheets.py` | results as CSV for Excel |
| `step6_build_website.py` | rebuilds the website in `docs/` |
| `step7_investigate.py` | canal test, offset periods, gauge vs. hand measurements |

## Setup from a terminal

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

## Findings on the real river (2022–2025)

**Data quality.** 927,484 fifteen-minute readings checked; 43,322 (4.7%) flagged.
- Upper Hudson discharge: 4.8% flagged, mostly a 4,190-reading stuck stretch and 2,320 estimated values.
- Green Island discharge: for most of 2024 USGS published only 2-hourly *estimated* values, so most of 2024 drops out of the balance and the models.
- Mohawk conductance and temperature sensors were stuck or spiking 29% and 22% of the time.

**Does the balance close?** On a typical day, yes: median daily gap +5.1%. Hour to hour it is
noisier (13.5% median absolute), consistent with dam and lock operations releasing water in pulses.

**The canal does not explain the gap.** The Barge Canal diverts Mohawk water around the Cohoes
gauge and returns it to the Hudson above Green Island, so it was the obvious candidate. Tested against the actual navigation seasons (2022–2025), the gap in cfs is the same with the
canal open (591 cfs) as closed (639 cfs), and moved the expected way at only 3 of 5 openings and
closings. It only looked seasonal in percent because summer flows are low (6,640 vs 18,440 cfs).

**What it actually is: a year-long step.** From December 2022 to November 2023 the gap jumps to a
median of 2,597 cfs, then drops back; outside that stretch it is 99 cfs (+0.7%). An offset that
switches on and off regardless of season or flow points to a calibration problem at one gauge.
`step7_investigate.py` compares each gauge with the USGS field crews' hand measurements to find
which one. (A smaller rise in autumn 2025 is mostly provisional data and may be revised.)

**Forecasting** (walk-forward, skill = error removed vs. "no change"):

| Horizon | Mass balance | Ridge | Boosted trees | Balance + trees |
|---|---|---|---|---|
| 6 h  | −32% | −3% | **+13%** | +7% |
| 12 h | −15% | +4% | **+20%** | +17% |
| 24 h | −17% | **+17%** | +14% | +11% |

- The pure mass-balance forecast loses to "no change": the calibration offset and lock operations
  make it biased. Trees learn around that; ridge extrapolates better at 24 h.
- Upstream gauges are what make it work: without them, skill drops to about zero.
- Rainfall adds nothing at 6–12 h but helps at 24 h (+14% vs +4% without it), matching the travel time from the basins.
- 90% prediction intervals covered only ~72% of outcomes before calibration and ~90% after conformal calibration.

**Anomaly detection.** The rules flag 12,960 hours, most of them sustained-imbalance hours from the
balance check; the Isolation Forest adds 82 hours of its own, short episodes worth inspecting by hand.

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
hudson/investigate.py  canal test, offset periods, calibration check against hand measurements
hudson/website.py      builds the project website from the latest results
hudson/report.py       validation charts and report
hudson/ml_report.py    forecasting and anomaly charts and report
hudson/synthetic.py    simulated river with known faults, for tests and the demo
run.py                 command-line entry point
tests/                 checks every validator, the leakage guarantees, and the models
```
