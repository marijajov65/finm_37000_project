"""Compare implied and native books on touch, size and width.

Same shape as ``run_implied_quotes.py`` -- it implies both directions on the
cached ES sessions:

  * ``implied_outright.implied_back_book``  - front + spread -> back
  * ``implied_spread.implied_spread_book``  - front + back   -> spread

but asks a different set of questions of the result. ``run_implied_quotes``
reports how far the implied quote is from the native one in ticks; here the
comparison is split into the three states a synthetic quote can be in
against the book it is competing with:

1. **Same touch.** Both sides match the native price exactly. The price is
   not the differentiator, so the question becomes size: which book shows
   more, and how often does the implied one?
2. **Better.** At least one side is strictly inside the native touch, so
   the implied quote would set a new best price on that side.
3. **Width.** The average bid-ask of each book over the window, in ticks
   and in price, regardless of the state above.

The three are reported per direction and per session. Sizes are compared
only in state 1 on purpose: comparing depth across books quoting different
prices measures nothing.

Window and instrument come from ``analysis_config`` and are all overridable
per run -- ``--days``, ``--start-utc``, ``--minutes``, ``--product``,
``--front-month``. Anything already in ``data_cache/`` is served from there;
a window that is not is fetched from Databento, which needs a key at
``~/.databento_api_key``. Pass ``--offline`` to make a cache miss an error
instead.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis_config import (
    BOOK_LEVELS,
    DAYS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)
from cached_market_data_fetcher import CachedMarketDataFetcher
from implied_common import align_books, infer_spread_convention, top_of_book
from implied_outright import implied_back_book
from implied_spread import implied_spread_book

#: Prices are compared after both books have been snapped to the same tick
#: grid, so an exact match is exact up to float representation only.
_PX_TOL = 1e-7


def compare_books(
    implied: pd.DataFrame, native_book: pd.DataFrame, tick: float
) -> pd.DataFrame:
    """Align an implied quote against the instrument's own book.

    Returns one row per event on the shared (forward filled) index, with the
    per-row flags the three headline statistics are averaged over:

    ``same_touch``
        both sides at the identical price.
    ``implied_better``
        either side strictly inside the native touch -- a bid above the
        native bid or an ask below the native ask. Strict: a match is not an
        improvement, and the two flags are mutually exclusive by
        construction.
    ``implied_more_size`` / ``native_more_size``
        total displayed quantity (bid + ask) of one book against the other.
        Only meaningful where ``same_touch`` holds, so both are masked to
        NaN elsewhere rather than set False -- a False would otherwise be
        averaged in as a loss when the row simply does not apply.

    Widths are carried through in ticks for the averages; NaN on either
    side (a one-sided book) propagates and drops out of the means.
    """
    native = top_of_book(native_book)
    aligned = align_books(implied=implied, native=native)
    imp, nat = aligned["implied"], aligned["native"]

    bid_match = np.isclose(imp["bid_px"], nat["bid_px"], atol=_PX_TOL, equal_nan=False)
    ask_match = np.isclose(imp["ask_px"], nat["ask_px"], atol=_PX_TOL, equal_nan=False)
    same_touch = pd.Series(bid_match & ask_match, index=imp.index)

    # Strictly inside on at least one side. The tolerance keeps a match from
    # registering as an improvement on float dust alone.
    bid_better = imp["bid_px"] > nat["bid_px"] + _PX_TOL
    ask_better = imp["ask_px"] < nat["ask_px"] - _PX_TOL
    implied_better = bid_better | ask_better

    imp_size = imp["bid_sz"] + imp["ask_sz"]
    nat_size = nat["bid_sz"] + nat["ask_sz"]
    # Size only decides anything when the prices tie; elsewhere it is not a
    # comparison, it is two different quotes.
    more = imp_size > nat_size
    less = imp_size < nat_size

    return pd.DataFrame(
        {
            "same_touch": same_touch,
            "implied_better": implied_better,
            "implied_more_size": more.where(same_touch),
            "native_more_size": less.where(same_touch),
            "implied_size": imp_size.where(same_touch),
            "native_size": nat_size.where(same_touch),
            "implied_width_ticks": imp["width"] / tick,
            "native_width_ticks": (nat["ask_px"] - nat["bid_px"]) / tick,
        },
        index=imp.index,
    )


def summarize(cmp: pd.DataFrame, tick: float) -> dict:
    """Reduce the per-event frame to the reported statistics.

    Percentages are shares of aligned updates, not of wall-clock time: the
    index is event driven, so every row is one change in one of the books.
    The size shares are conditional on ``same_touch`` -- they are shares of
    the matched rows, and the two plus the ties sum to 100%.
    """
    same = cmp["same_touch"]
    n_same = int(same.sum())

    return {
        "updates": len(cmp),
        "same_touch_pct": float(same.mean() * 100.0),
        "same_touch_n": n_same,
        # Conditional on a matched touch; NaN (rather than 0) when the two
        # books never agree on a price in this window.
        "implied_more_size_pct": float(cmp["implied_more_size"].mean() * 100.0)
        if n_same
        else float("nan"),
        "native_more_size_pct": float(cmp["native_more_size"].mean() * 100.0)
        if n_same
        else float("nan"),
        "avg_implied_size": float(cmp["implied_size"].mean()) if n_same else float("nan"),
        "avg_native_size": float(cmp["native_size"].mean()) if n_same else float("nan"),
        "implied_better_pct": float(cmp["implied_better"].mean() * 100.0),
        "avg_implied_width_ticks": float(cmp["implied_width_ticks"].mean()),
        "avg_native_width_ticks": float(cmp["native_width_ticks"].mean()),
        "avg_implied_width_px": float(cmp["implied_width_ticks"].mean()) * tick,
        "avg_native_width_px": float(cmp["native_width_ticks"].mean()) * tick,
    }


def _print_summary(name: str, stats: dict) -> None:
    tied = 100.0 - stats["implied_more_size_pct"] - stats["native_more_size_pct"]
    print(
        f"  {name}\n"
        f"    moments compared        {stats['updates']:,}"
        f"   (every time either book moved)\n"
        f"    same touch (both sides) {stats['same_touch_pct']:.2f}%"
        f"  ({stats['same_touch_n']:,} moments)\n"
        f"      implied shows more    {stats['implied_more_size_pct']:.2f}%"
        f"   real {stats['native_more_size_pct']:.2f}%"
        f"   equal {tied:.2f}%\n"
        f"      avg size bid+ask      implied {stats['avg_implied_size']:.1f}"
        f"   real {stats['avg_native_size']:.1f}\n"
        f"    implied better (either) {stats['implied_better_pct']:.2f}%\n"
        f"    avg width               implied {stats['avg_implied_width_ticks']:.2f} ticks"
        f" ({stats['avg_implied_width_px']:.4f})"
        f"   real {stats['avg_native_width_ticks']:.2f} ticks"
        f" ({stats['avg_native_width_px']:.4f})"
    )


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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    session_length = pd.Timedelta(minutes=args.minutes)

    # client=None is enough on a warm cache -- CachedMarketDataFetcher only
    # reaches for Databento on a miss, where it then fails on the missing
    # client. Building the real client is what lets an uncached window work.
    fetcher = (
        CachedMarketDataFetcher(client=None, levels=BOOK_LEVELS)
        if args.offline
        else CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    )
    spec = fetcher.fetch_calendar_spread_contract_specification(
        args.front_month, args.product
    )
    outright_tick = float(spec.outright_tick_size)
    spread_tick = float(spec.spread_tick_size)

    print(
        f"=== Implied vs native: {spec.spread_symbol} "
        f"({spec.front_symbol} / {spec.back_symbol}) ===\n"
        f"  tick: outright {spec.outright_tick_size}, spread {spec.spread_tick_size}"
    )

    legs = None
    for day in args.days:
        t0 = pd.Timestamp(f"{day}T{args.start_utc}", tz="UTC")
        t1 = t0 + session_length
        try:
            data = fetcher.fetch_calendar_spread_data(
                args.front_month, args.product, t0, t1
            )
        except AttributeError as exc:
            # A cache miss under --offline reaches for the client that isn't
            # there. Say what actually went wrong rather than surfacing
            # ``'NoneType' object has no attribute 'timeseries'``.
            if not args.offline:
                raise
            raise SystemExit(
                f"{day} {args.start_utc} +{args.minutes:g}min is not in "
                f"data_cache/; drop --offline to fetch it from Databento."
            ) from exc

        if legs is None:
            # Inferred once and held fixed, so the implied books for the rest
            # of the run never read the native book they are benchmarked
            # against. The symbol will not tell you the sign: ESM6-ESU6 is
            # back - front, ZSF6-ZSH6 is front - back.
            legs = infer_spread_convention(data.front, data.back, data.spread)
            print(f"  inferred convention: spread = {legs.name}")

        print(f"\n{day} {args.start_utc}-{t1:%H:%M:%S} UTC")

        back = implied_back_book(data, spec, legs=legs)
        _print_summary(
            f"implied {spec.back_symbol}  <- front + spread",
            summarize(compare_books(back, data.back, outright_tick), outright_tick),
        )

        spread = implied_spread_book(data, spec, legs=legs)
        _print_summary(
            f"implied {spec.spread_symbol}  <- front + back",
            summarize(compare_books(spread, data.spread, spread_tick), spread_tick),
        )


if __name__ == "__main__":
    main()
