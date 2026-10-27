"""Gauges, parameters and thresholds for the Hudson / Mohawk mass balance.

Everything you might want to tune lives here, so the analysis code
never hard-codes a site ID or a magic number.
"""

# --- Gauges -----------------------------------------------------------------
# The balance point is where the Mohawk joins the Hudson near Troy, which is
# still above the tidal estuary (the Troy dam is the head of tide).
#
#   Upper Hudson (01335754) ──┐
#                             ├──> Hudson at Green Island (01358000)
#   Mohawk at Cohoes (01357500)┘
#
# Verify each ID on https://waterdata.usgs.gov before a big pull, and run
# `python run.py inventory` to see which parameters each site actually has.
SITES = {
    "upper_hudson": {
        "id": "USGS-01335754",
        "name": "Hudson R above Lock 1 nr Waterford",
        "role": "inflow",
    },
    "mohawk": {
        "id": "USGS-01357500",
        "name": "Mohawk R at Cohoes",
        "role": "inflow",
        # Flow here is measured AFTER water is diverted into the Barge Canal
        # at Crescent Dam, so the balance is not expected to close perfectly.
    },
    "green_island": {
        "id": "USGS-01358000",
        "name": "Hudson R at Green Island",
        "role": "outflow",
    },
}

INFLOWS = [k for k, v in SITES.items() if v["role"] == "inflow"]
OUTFLOW = next(k for k, v in SITES.items() if v["role"] == "outflow")

# --- USGS parameter codes -----------------------------------------------------
PARAMS = {
    "00060": "discharge_cfs",
    "00065": "gage_height_ft",
    "00095": "spec_cond_uScm",   # specific conductance at 25 C
    "00300": "do_mgL",           # dissolved oxygen
    "00010": "temp_C",           # water temperature
}
FLOW = "discharge_cfs"

# --- Cleaning thresholds ------------------------------------------------------
GRID = "15m"                    # USGS continuous data is nominally 15-minute
FLATLINE_STEPS = 16             # 16 identical readings = 4 hours stuck
SPIKE_Z = 8.0                   # robust z-score on step changes
# Qualifiers that mean "this reading is not a real measurement". The new USGS
# API spells them out ("ESTIMATED"); the legacy service used short codes
# ("e", "Ice", "Eqp"). Both are matched, case-insensitively.
BAD_QUALIFIERS = ("ICE", "EQUIP", "EQP", "BACKWATER", "BKW", "MAINT", "MNT", "MALFUNCTION")
# ESTIMATED values were filled in by USGS staff, often during ice or a broken
# sensor. They're flagged separately so the report can count them, and kept
# out of model training. (Real example: Green Island reported only 2-hourly
# ESTIMATED values for most of 2024.)
ESTIMATED_QUALIFIERS = ("ESTIMATED",)

# --- Balance thresholds -------------------------------------------------------
MAX_LAG_HOURS = 24              # search window for travel time
BALANCE_TOL_PCT = 10.0          # |residual| above this is flagged
PERSIST_HOURS = 6               # ...but only if it lasts this long

# --- Rainfall (Open-Meteo historical archive, free, no key) ---------------------
# A few points inside each drainage basin; their hourly rain is averaged.
RAIN_POINTS = {
    "upper_hudson": [(43.31, -73.64), (43.65, -73.95), (43.10, -73.58)],  # Glens Falls, Adirondacks, Saratoga
    "mohawk": [(43.10, -75.23), (42.94, -74.19), (43.21, -75.46)],        # Utica, Amsterdam, Rome
}

# --- Forecasting --------------------------------------------------------------
HORIZONS = [6, 12, 24]          # hours ahead to forecast Green Island flow
N_FOLDS = 4                     # walk-forward test blocks
INITIAL_TRAIN_FRAC = 0.5        # first half of the record is always training
STORM_QUANTILE = 0.8            # flows above this (training) quantile count as "storm"
INTERVAL = (0.05, 0.95)         # 90% prediction interval

# --- Anomaly detection ----------------------------------------------------------
ANOMALY_CONTAMINATION = 0.02    # expected fraction of anomalous hours

# --- Chemistry thresholds -----------------------------------------------------
DO_SAT_LOW_PCT = 20.0           # below this is suspicious in a flowing river
DO_SAT_HIGH_PCT = 200.0         # above this is almost certainly a sensor problem
