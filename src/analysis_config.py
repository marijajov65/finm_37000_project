"""Fetch inputs shared by the analysis and the scout.

Defaults only -- both runners
accept --product/--front-month/--days/--minutes to override per run.
"""

from __future__ import annotations

import pandas as pd

# ES M6 = June 2026 front, ESU6 = Sep 2026 back; expiries 06-19 / 09-18.
PRODUCT = "ES"
FRONT_MONTH = "M6"

# mbp-1 = top of book (Databento allows 1 or 10; 10 for depth, larger pull).
BOOK_LEVELS = 1

# 5-minute window at 16:00 UTC. Longer gives the spread more variation but is a
# fresh, larger fetch.
SESSION_START_UTC = "16:00:00"
SESSION_LENGTH = pd.Timedelta(seconds=300)

# Roll-period days before the 06-19 expiry (busiest by spread activity, per the
# scout). Add 06-09/06-10 for the quiet mid-cycle contrast.
DAYS = ["2026-06-12", "2026-06-15", "2026-06-16"]
