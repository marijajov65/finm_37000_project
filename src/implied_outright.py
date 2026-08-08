"""Quote either outright from the spread and the other outright ("implied out").

Write the spread's definition::

    S = c_F * F + c_B * B

and solve it for whichever leg you are missing::

    B = (-c_F/c_B) * F + (1/c_B) * S        target = "back"
    F = (-c_B/c_F) * B + (1/c_F) * S        target = "front"

Both are the same one-line rearrangement. Which side of each book gets read
then follows from the coefficient signs, via
``implied_common.synthetic_side``.

The side pairing is a property of the leg, not the symbol
---------------------------------------------------------
It is tempting to think the pairing is fixed by the product's quoting
convention. It is not -- it is fixed by **whether the leg you are solving
for sits on the positive or negative side of that spread**, and the same
product gives you both. The soybean snapshot shows it inside one convention
(everything there is nearby - deferred):

    implying ZSF6 from ZSX5 and ZSX5-ZSF6
        S = X5 - F6, so F6 is the NEGATIVE leg:  F6 = X5 - S
        F6_bid = X5_bid - S_ask = 1008.50 - (-17.75) = 1026.25   <- spread ASK

    implying ZSF6 from ZSH6 and ZSF6-ZSH6
        S = F6 - H6, so F6 is the POSITIVE leg:  F6 = H6 + S
        F6_bid = H6_bid + S_bid = 1041.75 + (-15.50) = 1026.25   <- spread BID

Same product, same naming convention, same target instrument, same side of
that target -- and one route reads the spread's ask while the other reads
its bid. Both land on 1026.25, which is what keeps the curve
arbitrage-free. So a module that only ever solves for one leg can only ever
produce one of the two pairings, and will silently miss the other route.

Both sides of every component matter
------------------------------------
Across the two sides of the implied quote, every component book is read on
*both* of its sides -- the bid of the target uses one set, the ask the
other. There is no reduction where a component contributes only its bid or
only its ask. Anything that touches only one side of an input book is not
an implied quote.

Width
-----
Subtracting the two sides gives the sum of the (scaled) component widths, in
every case::

    implied width = |c_other/c_target| * other_width + |1/c_target| * S_width

Two one-tick markets imply a two-tick market. That is why an implied quote
rarely sets the top of book on a liquid outright and matters most on the
deferred, where the direct book is thin; it also means an implied quote can
never be crossed, since each component contributes a non-negative width.

Tick grid
---------
This direction is where rounding bites. On ES the outright tick is 0.25 but
the spread tick is 0.05, so the combination is a multiple of 0.05 that
usually is not a multiple of 0.25. It is moved to the nearest tradeable tick
*away* from the mid (offers up, bids down) and the fraction given up is
reported as ``*_round_edge`` rather than discarded -- on the 0.25-vs-0.05
grid it averages roughly 0.6 ticks, about $7.50 a lot.

Public API
----------
``imply_leg_quote``     - pure arithmetic; solves for "front" or "back".
``implied_back_book``   - target="back",  wired to a ``CalendarSpreadData``.
``implied_front_book``  - target="front", wired to a ``CalendarSpreadData``.

Use ``implied_common.combine_routes`` to reduce several routes to the single
best implied quote.
"""

from __future__ import annotations

from typing import Literal, Optional

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
IMPLIED_BACK_COLUMNS = QUOTE_COLUMNS

Target = Literal["front", "back"]


def leg_coefficients(legs: SpreadLegs, target: Target) -> tuple[float, float]:
    """Coefficients ``(on the known leg, on the spread)`` for ``target``.

    From ``S = c_F F + c_B B``, solving for one leg leaves the other leg
    weighted ``-c_other/c_target`` and the spread weighted ``1/c_target``.

        legs              target    (known, spread)   meaning
        BACK_MINUS_FRONT  back      ( 1,  1)          B = F + S
        BACK_MINUS_FRONT  front     ( 1, -1)          F = B - S
        FRONT_MINUS_BACK  back      ( 1, -1)          B = F - S
        FRONT_MINUS_BACK  front     ( 1,  1)          F = B + S

    Note the spread's coefficient flips with the target under a *fixed*
    convention: that is the "same product, both pairings" point above.
    """
    if target == "back":
        c_target, c_other = legs.back_coef, legs.front_coef
    elif target == "front":
        c_target, c_other = legs.front_coef, legs.back_coef
    else:
        raise ValueError(f"target must be 'front' or 'back', got {target!r}")
    return (-c_other / c_target, 1.0 / c_target)


def imply_leg_quote(
    known_leg: pd.DataFrame,
    spread: pd.DataFrame,
    outright_tick: float,
    legs: SpreadLegs,
    target: Target,
) -> pd.DataFrame:
    """Implied quote for one outright leg, from the other leg and the spread.

    ``known_leg`` and ``spread`` are top-of-book frames (``bid_px``,
    ``bid_sz``, ``ask_px``, ``ask_sz``) sharing an index -- see
    ``implied_common.top_of_book`` and ``align_books``. Kept separate from
    the book-level entry points so the arithmetic can be tested on
    hand-written quotes without a fetcher.

    ``target`` names the leg being quoted; ``known_leg`` is the other one.
    ``legs`` says how the spread is composed and has no default, because the
    wrong one is silent.

    Returns ``QUOTE_COLUMNS``. NaN propagates to whichever side of the
    implied quote actually needed the missing price -- which, when the
    spread carries a negative coefficient, is the *opposite* side of the
    spread book.
    """
    if not known_leg.index.equals(spread.index):
        raise ValueError("known_leg and spread quotes must share an index; align them first")

    known_coef, spread_coef = leg_coefficients(legs, target)
    return assemble_quote(
        [(known_coef, known_leg), (spread_coef, spread)], tick=outright_tick
    )


def imply_back_quote(
    front: pd.DataFrame,
    spread: pd.DataFrame,
    outright_tick: float,
    legs: SpreadLegs,
) -> pd.DataFrame:
    """``imply_leg_quote`` with ``target="back"``."""
    return imply_leg_quote(front, spread, outright_tick, legs, target="back")


def imply_front_quote(
    back: pd.DataFrame,
    spread: pd.DataFrame,
    outright_tick: float,
    legs: SpreadLegs,
) -> pd.DataFrame:
    """``imply_leg_quote`` with ``target="front"``."""
    return imply_leg_quote(back, spread, outright_tick, legs, target="front")


def _implied_leg_book(
    data: CalendarSpreadData,
    spec: CalendarSpreadContractSpec,
    legs: Optional[SpreadLegs],
    target: Target,
) -> pd.DataFrame:
    if legs is None:
        legs = infer_spread_convention(data.front, data.back, data.spread)
    known = data.front if target == "back" else data.back
    aligned = align_books(known=top_of_book(known), spread=top_of_book(data.spread))
    return imply_leg_quote(
        aligned["known"],
        aligned["spread"],
        outright_tick=float(spec.outright_tick_size),
        legs=legs,
        target=target,
    )


def implied_back_book(
    data: CalendarSpreadData,
    spec: CalendarSpreadContractSpec,
    legs: Optional[SpreadLegs] = None,
) -> pd.DataFrame:
    """Implied deferred-leg quote over the window, from ``data.front`` and
    ``data.spread``.

    The two source books carry independent event timestamps, so the result
    is indexed on their union, forward filled: the implied quote updates
    exactly when one of its inputs updates, starting once both have
    published.

    ``legs=None`` infers the convention, which is the only step that reads
    ``data.back``. Pass ``legs`` explicitly to keep the deferred book out of
    the calculation entirely -- necessary if the result is to be an
    independent benchmark for that book.
    """
    return _implied_leg_book(data, spec, legs, target="back")


def implied_front_book(
    data: CalendarSpreadData,
    spec: CalendarSpreadContractSpec,
    legs: Optional[SpreadLegs] = None,
) -> pd.DataFrame:
    """Implied front-leg quote over the window, from ``data.back`` and
    ``data.spread`` -- the mirror route, and the one that reads the opposite
    side of the spread book.

    ``legs=None`` infers the convention, which is the only step that reads
    ``data.front``.
    """
    return _implied_leg_book(data, spec, legs, target="front")
