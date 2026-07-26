"""Shared inputs for the Issue #11 analysis scripts — single source of truth.

These are the fetch parameters that MUST agree across
``run_relationship_analysis.py`` and ``scout_spread_activity.py``. The on-disk
Databento cache is keyed by symbol + schema + window, so if any of these drift
between scripts the scripts stop reusing each other's cached data and trigger
duplicate metered fetches. Change them here, once, and every script stays
cache-aligned.

Script-specific knobs (sampling grids, output dirs, scan/probe ranges) stay in
their own scripts — only the cache-key-determining inputs and the canonical
analysis days live here.
"""

from __future__ import annotations

import pandas as pd

# --- Instrument -------------------------------------------------------------
# Front leg; the back leg is the next quarter. ES M6 = June 2026 (front),
# ESU6 = Sep 2026 (back); expiries 2026-06-19 / 2026-09-18.
PRODUCT = "ES"
FRONT_MONTH = "M6"

# --- Data depth -------------------------------------------------------------
# mbp-1 = top of book. Databento supports only 1 or 10; use 10 for depth-of-book
# liquidity (a larger, more expensive pull).
BOOK_LEVELS = 1

# --- Window (CACHE-KEY-DETERMINING) -----------------------------------------
# 5 minutes from 16:00 UTC (10:00 AM CT, the README snapshot hour). Kept short so
# the first fetch is cheap and fine grids stay small. Trade-off: 300s is ample
# for the liquid front/back but thin for the spread (a few hundred spread trades
# even on a busy roll day), which is what keeps the spread hypotheses
# underpowered. Lengthening this (e.g. 1800s) is the lever for more spread
# variation — but it is a fresh, larger Databento fetch. Because it lives here,
# all three scripts change together and stay cache-aligned automatically.
SESSION_START_UTC = "16:00:00"
SESSION_LENGTH = pd.Timedelta(seconds=300)

# --- Primary analysis days --------------------------------------------------
# Roll-period run-up to the 2026-06-19 ESM6 expiry, chosen via
# scout_spread_activity.py by spread activity: 06-12/15/16 were the busiest
# (06-15 ~816 spread trades vs ~38-165 mid-cycle; 06-18 returned no data, the
# front having wound down into expiry). Mid-cycle 06-09/06-10 are the quiet
# baseline contrast — add them here to reproduce the quiet-vs-roll comparison.
DAYS = ["2026-06-12", "2026-06-15", "2026-06-16"]
