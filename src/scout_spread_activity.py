"""Rank candidate days by spread activity before committing a full analysis run.

The front/back legs are always busy; the spread is what decides whether a window
can test the spread hypotheses. Fetches the same window per day (cache-shared
with the analysis) and reports only the spread. Each new day is a metered fetch.

    uv run src/scout_spread_activity.py
"""

from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from relationship_analysis import spread_activity
from analysis_config import (
    BOOK_LEVELS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

# Run-up to the 2026-06-19 ESM6 expiry, plus the two mid-cycle baseline days.
# Wider than the analysis DAYS on purpose.
CANDIDATE_DAYS = [
    "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
    "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18",
]


def rank_spread_activity(fetcher, days=CANDIDATE_DAYS) -> pd.DataFrame:
    """Fetch each day and rank by spread trades, busiest first. Days with no
    data (holidays, post-expiry) are skipped."""
    rows = []
    for day in days:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        try:
            data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t0 + SESSION_LENGTH)
        except Exception as exc:  # noqa: BLE001 — one missing day shouldn't stop the scan
            print(f"  {day}: skipped ({type(exc).__name__})")
            continue
        rows.append({"day": day, **spread_activity(data)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("day").sort_values("spread_trades", ascending=False)


def top_active_days(n: int, days=CANDIDATE_DAYS) -> list[str]:
    """The n busiest days, busiest first. Fetches every day (metered)."""
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    return list(rank_spread_activity(fetcher, days).index[:n])


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    table = rank_spread_activity(fetcher)
    if table.empty:
        print("no data for any candidate day")
        return
    print("Ranked by spread trades (busiest first):")
    print(table.to_string())


if __name__ == "__main__":
    main()
