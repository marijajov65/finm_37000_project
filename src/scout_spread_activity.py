"""Scout spread-instrument liveliness across candidate days (Issue #11 helper).

Before committing a full mbp-1 book pull to a set of days, this fetches each
day's spread window and reports only how much the SPREAD instrument moved and
traded. Use it to find the roll-period days worth analyzing — the front/back
legs are always busy, so the spread's activity is the thing that decides
whether a window can test the spread-vs-basis and lead-lag hypotheses.

Cheap relative to the full analysis: it fetches the same window per day but
only summarizes the spread (via ``relationship_analysis.spread_activity``), and
it reuses the disk cache, so days already pulled (e.g. 06-09, 06-10) cost
nothing. Still, each NEW day is a metered Databento fetch — that is the point,
to spend the minimum before picking the days for run_relationship_analysis.py.

Edit CANDIDATE_DAYS to the run-up to the ESM6 expiry (2026-06-19); the roll
typically peaks in the ~8 business days before the third Friday. Then run the
full analysis only on the busiest one or two.

Run:
    uv run src/scout_spread_activity.py
"""

from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from relationship_analysis import spread_activity

# Shared fetch inputs (kept identical across scripts via one module so cached
# windows are reused). See analysis_config.py.
from analysis_config import (  # noqa: E402
    BOOK_LEVELS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

#: Days to scout — the run-up to the ESM6 expiry on 2026-06-19. This scan list is
#: scout-specific (wider than the analysis DAYS in analysis_config); it includes
#: the two mid-cycle baseline days to show how quiet "not the roll" looks.
CANDIDATE_DAYS = [
    "2026-06-09",  # baseline (quiet)
    "2026-06-10",  # baseline (quiet)
    "2026-06-11",
    "2026-06-12",
    "2026-06-15",
    "2026-06-16",
    "2026-06-17",
    "2026-06-18",  # last session before 06-19 expiry
]


def rank_spread_activity(fetcher, days=CANDIDATE_DAYS) -> pd.DataFrame:
    """Fetch each day's window and rank by spread activity (busiest first).

    Returned frame is indexed by day with spread updates / mid-moves / trades /
    volume columns. Days with no data (holidays, post-expiry) are skipped.
    Reused by run_relationship_analysis.py for opt-in auto-selection.
    """
    rows = []
    for day in days:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        try:
            data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)
        except Exception as exc:  # noqa: BLE001 — a missing/holiday day shouldn't stop the scan
            print(f"  {day}: skipped ({type(exc).__name__}: {exc})")
            continue
        act = spread_activity(data)
        rows.append(
            {
                "day": day,
                "spread_updates": act["spread_updates"],
                "spread_mid_moves": act["spread_mid_moves"],
                "spread_trades": act["spread_trades"],
                "spread_volume": act["spread_volume"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["spread_updates", "spread_mid_moves", "spread_trades", "spread_volume"]
        )
    return pd.DataFrame(rows).set_index("day").sort_values("spread_trades", ascending=False)


def top_active_days(n: int, days=CANDIDATE_DAYS) -> list[str]:
    """Convenience: the ``n`` days with the most spread trades, busiest first.

    Builds its own fetcher and calls rank_spread_activity. NOTE: this fetches
    every day in ``days`` (metered on first pull), so callers should treat it as
    a re-scan, not a cheap lookup.
    """
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    ranked = rank_spread_activity(fetcher, days)
    return list(ranked.index[:n])


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    print(f"scouting {spec.spread_symbol} spread activity "
          f"({SESSION_START_UTC} UTC, {int(SESSION_LENGTH.total_seconds())}s windows)\n")

    table = rank_spread_activity(fetcher)
    if table.empty:
        print("no data fetched for any candidate day")
        return

    print("Ranked by spread trades (busiest first):")
    print(table.to_string())

    best = table.index[0]
    print(
        f"\nBusiest: {best} "
        f"({int(table.loc[best, 'spread_trades'])} trades, "
        f"{int(table.loc[best, 'spread_mid_moves'])} mid moves). "
        f"Run run_relationship_analysis.py on the top day(s)."
    )


if __name__ == "__main__":
    main()
