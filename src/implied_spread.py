"""Quote the calendar spread from its two outright legs ("implied in").

This direction needs no algebra -- the spread's definition *is* the
synthetic::

    S = c_F * F + c_B * B

so the coefficients go straight into ``implied_common.synthetic_side``. One
leg always carries a negative coefficient, so the two legs are always read
on **opposite** sides. Which leg is which depends on the convention.

Both conventions, worked
------------------------
**ES** (``BACK_MINUS_FRONT``: c_F = -1, c_B = +1)::

    S_ask = B_ask - F_bid          S_bid = B_bid - F_ask

Long the spread is long the back and short the front, so you pay the back's
offer and sell the front into its bid.

**ZS** (``FRONT_MINUS_BACK``: c_F = +1, c_B = -1) -- everything mirrors::

    S_ask = F_ask - B_bid          S_bid = F_bid - B_ask

Against the soybean snapshot (ZSF6 1026.25/1026.50, ZSH6 1041.75/1042.00,
native ZSF6-ZSH6 -15.50/-15.25)::

    S_bid = 1026.25 - 1042.00 = -15.75   (just under the native bid)
    S_ask = 1026.50 - 1041.75 = -15.25   (ties the native offer)

The commonest error in this direction is netting mid against mid, or the
same side against itself. That produces the spread's *fair value*, not a
quote: a price nobody will trade with you at, because it assumes you cross
neither leg.

Why this direction is the weak one
----------------------------------
Subtracting the two sides gives the sum of the leg widths, and that sum is
measured against a far finer grid than the legs trade on::

    implied width = |c_F| * F_width + |c_B| * B_width

Both ES legs might be one tick wide, but the spread implied out of them is
two outright ticks -- 0.50 -- while the spread instrument itself trades on a
0.05 tick. The native spread book can quote ten times finer than anything
its own legs can imply, and on the cached ES sessions the implied spread is
at or inside the native top of book under 1% of the time, against ~95% for
the deferred outright implied the other way.

That asymmetry is worth stating before any lead-lag claim: the deferred
outright genuinely can be quoted from front + spread at competitive width,
but the spread usually cannot be quoted competitively from its legs. The two
directions are not mirror images, and the implied engine linking them is why
a lag regression on the three mids peaks at zero -- the books are
mechanically tied, not merely correlated.

Tick grid
---------
Less of an issue here on ES: a difference of two prices that are both
multiples of 0.25 is a multiple of 0.25, hence already a multiple of the
0.05 spread tick. Rounding is still applied, conservatively (offers up, bids
down), because it is not free on every product -- any product whose spread
tick is coarser than its outright tick lands off-grid here, and silently
keeping an unshowable price would overstate implied liquidity.

Size
----
One lot of implied spread needs one lot of each leg, on opposite sides, so
the pairing is diagonal: under ES's convention the ask size pairs the back's
ask with the *front's bid*. Getting this backwards is easy and quiet -- the
numbers stay plausible, they are just the wrong book's depth. Deriving the
side from the coefficient's sign is what stops that happening by hand.

Public API
----------
``imply_spread_quote``  - pure arithmetic on aligned leg quotes.
``implied_spread_book`` - the same, wired to a ``CalendarSpreadData``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from implied_common import (
    QUOTE_COLUMNS,
    SpreadLegs,
    align_books,
    assemble_quote,
    infer_spread_convention,
    top_of_book,
)
from market_data_fetcher import CalendarSpreadContractSpec, CalendarSpreadData

#: Columns of the frames returned by this module.
IMPLIED_SPREAD_COLUMNS = QUOTE_COLUMNS


def imply_spread_quote(
    front: pd.DataFrame,
    back: pd.DataFrame,
    spread_tick: float,
    legs: SpreadLegs,
) -> pd.DataFrame:
    """Implied spread quote from already-aligned front and back leg quotes.

    ``front`` and ``back`` are top-of-book frames (``bid_px``, ``bid_sz``,
    ``ask_px``, ``ask_sz``) sharing an index -- see
    ``implied_common.top_of_book`` and ``align_books``. Kept separate from
    ``implied_spread_book`` so the arithmetic can be tested on hand-written
    quotes without a fetcher.

    ``legs`` says how the spread instrument is composed. It deliberately has
    no default: under the wrong convention every price comes out with the
    sign flipped and both sizes come from the wrong side of the wrong book,
    with nothing raised.

    Returns ``IMPLIED_SPREAD_COLUMNS``. The sides pair diagonally, so under
    ``BACK_MINUS_FRONT`` the implied *ask* is NaN if either the back's ask
    or the front's *bid* is missing.
    """
    if not front.index.equals(back.index):
        raise ValueError("front and back quotes must share an index; align them first")

    return assemble_quote(
        [(legs.front_coef, front), (legs.back_coef, back)], tick=spread_tick
    )


def implied_spread_book(
    data: CalendarSpreadData,
    spec: CalendarSpreadContractSpec,
    legs: Optional[SpreadLegs] = None,
) -> pd.DataFrame:
    """Implied spread quote over the whole window, from ``data.front`` and
    ``data.back``.

    Indexed on the union of the two leg books' event timestamps, forward
    filled, starting once both have published.

    ``legs=None`` infers the convention via ``infer_spread_convention``,
    which is the only step that reads ``data.spread``. Pass ``legs``
    explicitly to keep the native spread book out of the calculation
    entirely -- necessary if the result is to be a benchmark for it.
    """
    if legs is None:
        legs = infer_spread_convention(data.front, data.back, data.spread)

    aligned = align_books(
        front=top_of_book(data.front),
        back=top_of_book(data.back),
    )
    return imply_spread_quote(
        aligned["front"],
        aligned["back"],
        spread_tick=float(spec.spread_tick_size),
        legs=legs,
    )
