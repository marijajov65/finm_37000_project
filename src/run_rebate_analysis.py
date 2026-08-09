"""Issue #9 -> Issue #10 end to end, on a window you choose.

``run_pivotality_inputs.py`` produces the P&L sample set and writes it to
``results/pivotality/``; ``rebate_pivotality_analyzer.py`` consumes a
``List[pd.Series]`` of those samples and reports what the exchange would
have to give back. Nothing connects them -- the analyzer has no import of
the producer, and the producer's window is hard-coded. This script is that
connection: it replays the sweep in memory for the sessions you name and
feeds the result straight into the analyzer.

The knobs
---------
``--fill-rate`` is how often our resting quote is the one that trades when a
trade happens at our price; ``--inventory-limit`` is how many spread contracts
we carry before legging the excess out. Fills are random, so one replay is a
single draw of the P&L rather than its expectation -- ``--n-runs`` replays and
averages.

Raising ``--fill-rate`` is not an incentive the exchange can hand out: ES
matches FIFO on Globex, and CME's market-maker programs move the *fee*, not the
allocation. It describes a faster or better-placed quote, nothing more.

Window arguments match ``run_strategy_analysis.py`` (``--days``,
``--start-utc``, ``--minutes``, ``--product``, ``--front-month``,
``--offline``), so both analyses can be pointed at the same session.

Run:
    uv run src/run_rebate_analysis.py
    uv run src/run_rebate_analysis.py --days 2026-06-09 --minutes 1 --offline
"""

from __future__ import annotations

import argparse

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from fee_schedule import get_fee_rates
from legging_cost_calculator import LeggingCostCalculator
from rebate_pivotality_analyzer import RebatePivotalityAnalyzer
from run_pivotality_inputs import (
    BASE_SEED,
    BOOK_LEVELS,
    DAYS,
    FEE_TIER,
    FRONT_MONTH,
    POOLED_LABEL,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
    as_pnl_series,
    load_sessions,
    sweep,
)

#: Our quote wins the trade half the time it is at the touch.
DEFAULT_FILL_RATE = 0.50

#: Spread contracts carried before the excess is legged out.
DEFAULT_INVENTORY_LIMIT = 5.0

#: Fewer than run_pivotality_inputs' 50: this runs one configuration, not 36,
#: and the mean over 20 runs is already stable. Raise it for a headline number.
DEFAULT_N_RUNS = 20


def _parse_args() -> argparse.Namespace:
    default_minutes = SESSION_LENGTH.total_seconds() / 60.0
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--product", default=PRODUCT)
    p.add_argument("--front-month", default=FRONT_MONTH)
    p.add_argument("--days", nargs="+", default=DAYS,
                   help="YYYY-MM-DD, one or more")
    p.add_argument("--start-utc", default=SESSION_START_UTC,
                   help="session start, HH:MM:SS UTC")
    p.add_argument("--minutes", type=float, default=default_minutes,
                   help="session length in minutes")
    p.add_argument("--offline", action="store_true",
                   help="serve from data_cache/ only; error on a cache miss")
    p.add_argument("--fill-rate", "--p", type=float, default=DEFAULT_FILL_RATE,
                   dest="fill_rate",
                   help="0-1: chance our resting quote is the one filled")
    p.add_argument("--inventory-limit", "--max-position", type=float,
                   default=DEFAULT_INVENTORY_LIMIT, dest="inventory_limit",
                   help="spread contracts held before the excess is legged out")
    p.add_argument("--n-runs", "--n-seeds", type=int, default=DEFAULT_N_RUNS,
                   dest="n_runs",
                   help="simulations averaged (each is one random fill path)")
    p.add_argument("--fee-tier", default=FEE_TIER)
    p.add_argument("--daily-fixed-costs", type=float, default=0.0,
                   help="infrastructure cost per day the strategy must clear")
    p.add_argument("--session", default=POOLED_LABEL,
                   help=f"a day from --days, or {POOLED_LABEL!r} for the pooled run")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    session_length = pd.Timedelta(minutes=args.minutes)

    fetcher = (
        CachedMarketDataFetcher(client=None, levels=BOOK_LEVELS)
        if args.offline
        else CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    )
    spec = fetcher.fetch_calendar_spread_contract_specification(
        args.front_month, args.product
    )
    fees = get_fee_rates(spec.product_code, args.fee_tier)

    end_utc = (
        pd.Timestamp(f"2000-01-01T{args.start_utc}") + session_length
    ).strftime("%H:%M:%S")
    print(
        f"=== Rebate pivotality: {spec.spread_symbol} "
        f"({spec.front_symbol} / {spec.back_symbol}) ===\n"
        f"\n"
        f"Simulation parameters\n"
        f"  sessions                 {', '.join(args.days)}\n"
        f"  window (UTC)             {args.start_utc}-{end_utc} "
        f"({args.minutes:g} min each)\n"
        f"  chance of getting filled {args.fill_rate:.0%}"
        f"   (when a trade happens at our quote)\n"
        f"  inventory limit          {args.inventory_limit:g} spread contracts"
        f"   (excess is legged out)\n"
        f"  simulations averaged     {args.n_runs}\n"
        f"  exchange fees ({args.fee_tier})\n"
        f"      passive fill         ${fees.passive_fee:.2f} per spread contract\n"
        f"      legging out          ${fees.aggressive_fee:.2f} per spread contract\n"
        f"  daily fixed costs        ${args.daily_fixed_costs:,.2f}"
    )

    try:
        sessions, _ = load_sessions(
            fetcher,
            LeggingCostCalculator(spec),
            product=args.product,
            front_month=args.front_month,
            days=args.days,
            start_utc=args.start_utc,
            session_length=session_length,
            verbose=False,
        )
    except AttributeError as exc:
        # A cache miss under --offline reaches for the client that isn't
        # there; say so rather than surfacing the NoneType attribute error.
        if not args.offline:
            raise
        raise SystemExit(
            f"{args.start_utc} +{args.minutes:g}min on one of {args.days} is "
            "not in data_cache/; drop --offline to fetch it from Databento."
        ) from exc

    # Only the requested cell, not the full 36-cell grid.
    samples = sweep(
        sessions,
        args.days,
        fees,
        float(spec.contract_multiplier),
        p_grid=(args.fill_rate,),
        max_position_grid=(args.inventory_limit,),
        n_seeds=args.n_runs,
        base_seed=BASE_SEED,
    )

    cell = samples[
        (samples["session"] == args.session)
        & (samples["max_position"] == args.inventory_limit)
    ]
    if cell.empty:
        raise SystemExit(
            f"no samples for session={args.session!r}; expected one of "
            f"{args.days + [POOLED_LABEL]}"
        )

    analyzer = RebatePivotalityAnalyzer(
        current_passive_fee=fees.passive_fee,
        current_aggressive_fee=fees.aggressive_fee,
        daily_fixed_costs=args.daily_fixed_costs,
    )
    report = analyzer.generate_pivotality_report(as_pnl_series(cell))

    label = "pooled across days" if args.session == POOLED_LABEL else args.session
    print(f"\nResults ({label})")
    print(report.drop(columns="Flow Type").to_string(index=False))

    target = report[report["Scenario"] == "Target Break-Even"][
        "Passive Fee/Rebate"
    ].iloc[0]
    print(
        f"\nBreak-even passive fee: ${target:.4f} per spread contract"
        f"   (currently ${fees.passive_fee:.2f})"
    )


if __name__ == "__main__":
    main()
