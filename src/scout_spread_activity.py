"""Rank candidate days before a full analysis run.

Reports spread activity and the front's range/realized vol per day; rank by
--by trades (busiest spread) or --by vol (where the lead-lag question has
signal). Cache-shared with the analysis; each new day is a metered fetch.

    uv run src/scout_spread_activity.py
    uv run src/scout_spread_activity.py --by vol --minutes 30
"""

from __future__ import annotations

import argparse

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from relationship_analysis import front_volatility, spread_activity
from analysis_config import (
    BOOK_LEVELS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

# Run-up to the 2026-06-19 ESM6 expiry plus two mid-cycle baselines. Override
# with --days for a known high-vol session.
CANDIDATE_DAYS = [
    "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
    "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18",
]

SORT_KEYS = {"trades": "spread_trades", "vol": "front_rv_ticks"}


def rank_spread_activity(
    fetcher,
    days=CANDIDATE_DAYS,
    product: str = PRODUCT,
    front_month: str = FRONT_MONTH,
    start_utc: str = SESSION_START_UTC,
    length: pd.Timedelta = SESSION_LENGTH,
    sort_by: str = "trades",
) -> pd.DataFrame:
    """Fetch each day and rank by ``sort_by`` ('trades' or 'vol'). Days with no
    data (holidays, post-expiry) are skipped."""
    spec = fetcher.fetch_calendar_spread_contract_specification(front_month, product)
    rows = []
    for day in days:
        t0 = pd.Timestamp(f"{day}T{start_utc}", tz="UTC")
        try:
            data = fetcher.fetch_calendar_spread_data(front_month, product, t0, t0 + length)
        except Exception as exc:  # noqa: BLE001 — one missing day shouldn't stop the scan
            print(f"  {day}: skipped ({type(exc).__name__})")
            continue
        rows.append({"day": day, **spread_activity(data), **front_volatility(data, spec)})
    if not rows:
        return pd.DataFrame()
    key = SORT_KEYS[sort_by]
    return pd.DataFrame(rows).set_index("day").sort_values(key, ascending=False)


def top_active_days(
    n: int,
    days=CANDIDATE_DAYS,
    product: str = PRODUCT,
    front_month: str = FRONT_MONTH,
    start_utc: str = SESSION_START_UTC,
    length: pd.Timedelta = SESSION_LENGTH,
    sort_by: str = "trades",
) -> list[str]:
    """The n busiest (or most volatile) days, best first. Fetches every day (metered)."""
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    ranked = rank_spread_activity(fetcher, days, product, front_month, start_utc, length, sort_by)
    return list(ranked.index[:n])


def _parse_args() -> argparse.Namespace:
    default_minutes = SESSION_LENGTH.total_seconds() / 60.0
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", default=PRODUCT)
    p.add_argument("--front-month", default=FRONT_MONTH)
    p.add_argument("--days", nargs="+", default=CANDIDATE_DAYS)
    p.add_argument("--minutes", type=float, default=default_minutes)
    p.add_argument("--start-utc", default=SESSION_START_UTC)
    p.add_argument("--by", choices=SORT_KEYS.keys(), default="trades",
                   help="rank by spread 'trades' (default) or front 'vol'")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    table = rank_spread_activity(
        fetcher, args.days, args.product, args.front_month,
        args.start_utc, pd.Timedelta(minutes=args.minutes), args.by,
    )
    if table.empty:
        print("no data for any candidate day")
        return
    label = "spread trades" if args.by == "trades" else "front realized vol"
    print(f"Ranked by {label} (top first):")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
