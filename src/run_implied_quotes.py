"""Run the two implied-quoting modules on the cached ES demo sessions.

Wires ``CachedMarketDataFetcher`` (the same offline path
``run_pnl_pipeline.py`` uses) into:

  * ``implied_outright.implied_back_book``  - front + spread -> back
  * ``implied_spread.implied_spread_book``  - front + back   -> spread

and prints, per session, how the implied quote compares with the venue's own
book for the instrument being implied. Both directions are reported because
the point of splitting them into two modules is that they are *not* mirror
images: the deferred outright can usually be implied at a competitive width,
the spread usually cannot.

Runs offline against ``data_cache/`` (no API key needed once warm), matching
the windows ``demo_print_books.py`` / ``run_pnl_pipeline.py`` already cached.
"""

from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from implied_common import align_books, infer_spread_convention, top_of_book
from implied_outright import implied_back_book
from implied_spread import implied_spread_book

PRODUCT = "ES"
FRONT_MONTH = "M6"

DAYS = ["2026-06-09", "2026-06-10"]
SESSION_START_UTC = "16:00:00"
SESSION_LENGTH = pd.Timedelta(seconds=300)

BOOK_LEVELS = 1  # implied quoting reads top of book only


def compare_to_native(implied: pd.DataFrame, native_book: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Line up an implied quote against the instrument's own book.

    Both are forward filled onto their shared event index, then each side is
    measured in ticks of *improvement*: how much better the implied quote is
    than the native one. Positive means the implied would set a new best
    price; zero means it matches; negative means the native book is inside
    and the implied never shows.

        bid improvement = (implied_bid - native_bid) / tick
        ask improvement = (native_ask - implied_ask) / tick

    This is descriptive only -- it says which book is at the top, not which
    one moved first.
    """
    native = top_of_book(native_book)
    aligned = align_books(implied=implied, native=native)
    imp, nat = aligned["implied"], aligned["native"]
    return pd.DataFrame(
        {
            "bid_improve_ticks": (imp["bid_px"] - nat["bid_px"]) / tick,
            "ask_improve_ticks": (nat["ask_px"] - imp["ask_px"]) / tick,
            "implied_width_ticks": imp["width"] / tick,
            "native_width_ticks": (nat["ask_px"] - nat["bid_px"]) / tick,
            "implied_bid_sz": imp["bid_sz"],
            "native_bid_sz": nat["bid_sz"],
        }
    )


def _summarize(name: str, implied: pd.DataFrame, cmp: pd.DataFrame, tick: float) -> None:
    at_or_better = ((cmp["bid_improve_ticks"] >= 0) | (cmp["ask_improve_ticks"] >= 0)).mean()
    print(
        f"  {name}\n"
        f"    updates                 {len(implied):,}\n"
        f"    implied width (ticks)   median {cmp['implied_width_ticks'].median():.1f}"
        f"   native {cmp['native_width_ticks'].median():.1f}\n"
        f"    implied at/inside best  {at_or_better:.1%} of the time\n"
        f"    median size bid/ask     {implied['bid_sz'].median():.0f} / {implied['ask_sz'].median():.0f}\n"
        f"    tick rounding conceded  {implied['bid_round_edge'].mean() / tick:.2f} ticks (bid, mean)"
    )


def main() -> None:
    # client=None is fine on a warm cache: CachedMarketDataFetcher only
    # reaches for Databento on a cache miss. Swap in
    # CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS) to
    # fetch new windows.
    fetcher = CachedMarketDataFetcher(client=None, levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    outright_tick = float(spec.outright_tick_size)
    spread_tick = float(spec.spread_tick_size)

    print(
        f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})\n"
        f"  tick: outright {spec.outright_tick_size}, spread {spec.spread_tick_size}"
    )

    legs = None
    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)

        if legs is None:
            # Read the sign convention off the first session, then hold it
            # fixed. Inferred once rather than per-call so the implied quote
            # for the rest of the run never touches the book it is meant to
            # be a benchmark for. The symbol name will not tell you this:
            # ESM6-ESU6 is back - front, ZSF6-ZSH6 is front - back.
            legs = infer_spread_convention(data.front, data.back, data.spread)
            print(f"  inferred convention: spread = {legs.name}")

        print(f"\n{day} {SESSION_START_UTC}-{t1:%H:%M:%S} UTC")

        back = implied_back_book(data, spec, legs=legs)
        _summarize(
            f"implied {spec.back_symbol}  <- front + spread",
            back,
            compare_to_native(back, data.back, outright_tick),
            outright_tick,
        )

        spread = implied_spread_book(data, spec, legs=legs)
        _summarize(
            f"implied {spec.spread_symbol}  <- front + back",
            spread,
            compare_to_native(spread, data.spread, spread_tick),
            spread_tick,
        )


if __name__ == "__main__":
    main()
