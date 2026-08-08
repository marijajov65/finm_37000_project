"""Run the lead-lag & co-movement analysis over the configured days (issue #11).

Prints a one-block summary per day; --full adds the underlying tables. Offline
from the cached windows; an unseeded cache needs a Databento key. Defaults come
from analysis_config.py, overridable on the CLI.

    uv run src/run_relationship_analysis.py
    uv run src/run_relationship_analysis.py --days 2026-06-12 --minutes 30 --full
"""

from __future__ import annotations

import argparse

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from relationship_analysis import (
    change_corr,
    deferred_vs_implied,
    lead_lag_table,
    liquidity_metrics,
    resample_mids,
    spread_basis_sweep,
    spread_mid_diagnostic,
    trade_lead_lag_table,
)
from analysis_config import (
    BOOK_LEVELS,
    DAYS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

PRIMARY_FREQ = "500ms"
MAX_LAG = 40  # grid steps; 40 * 500ms = 20s


def _parse_args() -> argparse.Namespace:
    default_minutes = SESSION_LENGTH.total_seconds() / 60.0
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", default=PRODUCT)
    p.add_argument("--front-month", default=FRONT_MONTH)
    p.add_argument("--days", nargs="+", default=None)
    p.add_argument("--minutes", type=float, default=default_minutes)
    p.add_argument("--start-utc", default=SESSION_START_UTC)
    p.add_argument("--freq", default=PRIMARY_FREQ)
    p.add_argument("--max-lag", type=int, default=MAX_LAG)
    p.add_argument("--full", action="store_true", help="also print the full tables")
    return p.parse_args()


def _summary(data, spec, panel, freq, max_lag) -> None:
    """The five lines that carry the findings."""
    ll = lead_lag_table(panel, max_lag=max_lag)
    fb = ll[(ll.a == "front_mid") & (ll.b == "back_mid")].iloc[0]
    tll = trade_lead_lag_table(data, freq=freq, max_lag=max_lag)
    tfb = tll[(tll.a == "front_flow") & (tll.b == "back_flow")]
    tflow = f"{tfb.iloc[0]['peak_corr']:.2f}" if not tfb.empty else "n/a"

    impl = deferred_vs_implied(panel)
    moves = spread_mid_diagnostic(data, spec)["n_mid_moves"]
    sweep = spread_basis_sweep(data)
    rate = liquidity_metrics(data, spec)["update_rate_per_s"]

    print(f"  front<->back    lag {fb['peak_lag']}, corr {fb['peak_corr']:.2f} "
          f"(trade flow {tflow})")
    print(f"  deferred=f+s    R2={impl['r_squared']:.3f}  b_front={impl['beta_front']:.2f}  "
          f"b_spread={impl['beta_spread']:.2f}  ({moves} spread moves)")
    corrs = " / ".join(f"{c:.2f}" for c in sweep["zerolag_corr"])
    grids = " ".join(sweep["freq"])
    print(f"  spread<->basis  {corrs}  ({grids})")
    print(f"  updates/s       front {rate['front']:.0f}  back {rate['back']:.0f}  "
          f"spread {rate['spread']:.0f}")


def _full(data, spec, panel, freq, max_lag) -> None:
    """The tables behind the summary."""
    print("\n  Liquidity:")
    print(liquidity_metrics(data, spec).round(3).to_string())
    print(f"\n  Change correlations ({freq}):")
    print(change_corr(panel).round(3).to_string())
    print("\n  Lead-lag, quote mids:")
    print(lead_lag_table(panel, max_lag=max_lag).to_string(index=False))
    print("\n  Lead-lag, trade flow:")
    print(trade_lead_lag_table(data, freq=freq, max_lag=max_lag).to_string(index=False))
    print("\n  Spread vs basis by interval (peak_lag_s < 0 = basis leads):")
    print(spread_basis_sweep(data).to_string(index=False))


def main() -> None:
    args = _parse_args()
    session_length = pd.Timedelta(minutes=args.minutes)
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(args.front_month, args.product)
    print(f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})\n")

    days = args.days if args.days is not None else DAYS
    for day in days:
        t0 = pd.Timestamp(f"{day}T{args.start_utc}", tz="UTC")
        t1 = t0 + session_length
        data = fetcher.fetch_calendar_spread_data(args.front_month, args.product, t0, t1)
        panel = resample_mids(data, freq=args.freq)

        print(f"=== {day} {args.start_utc}..{t1:%H:%M:%S} UTC  "
              f"(front {len(data.front):,} / back {len(data.back):,} / "
              f"spread {len(data.spread):,}) ===")
        _summary(data, spec, panel, args.freq, args.max_lag)
        if args.full:
            _full(data, spec, panel, args.freq, args.max_lag)
        print()


if __name__ == "__main__":
    main()
