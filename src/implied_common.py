"""Shared primitives for the two implied-quoting modules.

``implied_outright`` (front + spread -> back) and ``implied_spread``
(front + back -> spread) are the two public modules; this holds what they
both need so neither depends on the other.

The one idea worth stating up front
-----------------------------------
An implied quote is a *synthetic instrument*::

    X = sum_i c_i * Y_i

and the only rule you need is: **to sell X you sell every component with a
positive coefficient and buy every component with a negative one.** So the
implied bid of X reads the ``bid`` of each positively-weighted component and
the ``ask`` of each negatively-weighted one::

    X_bid = sum_{c_i>0} c_i * Y_i_bid  +  sum_{c_i<0} c_i * Y_i_ask
    X_ask = sum_{c_i>0} c_i * Y_i_ask  +  sum_{c_i<0} c_i * Y_i_bid

Nothing about calendar spreads is special here; ``synthetic_side`` below is
just that formula, and both public modules are two lines of it.

Why this is not a detail
------------------------
Which side of the spread book you read depends on the sign of the spread's
coefficient, and **that sign is not something you can read off the symbol.**
Two real snapshots, both named ``LEG1-LEG2`` with the nearby leg first:

    ESM6-ESU6 quoted +60.78, with ESM6 7313.1 and ESU6 7373.7
        -> the spread is BACK - FRONT

    ZSF6-ZSH6 quoted -15.50/-15.25, with ZSF6 1026.4 and ZSH6 1041.9
        -> the spread is FRONT - BACK

Equity index and grains use opposite conventions under identical-looking
names. Assume ES's convention and run it on soybeans and the implied ZSH6
bid comes out 1010.75 against a native bid of 1041.75 -- wrong by 30.75, or
twice the spread level, with no exception and no NaN to warn you. That is
exactly the failure mode ``legging_cost_calculator``'s docstring warns about
("roughly twice the spread level ... check it before trusting the P&L").

So the convention is an explicit input (``SpreadLegs``), and
``infer_spread_convention`` will read it off the data rather than let anyone
guess.

numpy + pandas only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

#: Level-00 columns of a book frame, in (price, size) pairs per side.
TOP_BOOK_COLUMNS = ("bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00")

#: Columns of a top-of-book frame as returned by ``top_of_book``.
TOP_COLUMNS = ("bid_px", "bid_sz", "ask_px", "ask_sz")

Side = Literal["bid", "ask"]

#: Rounding a price onto the tick grid can only ever move it *away* from
#: the mid, never toward it. See ``round_to_tick``.
Direction = Literal["up", "down"]

#: Guards against float noise in ``price / tick`` (e.g. 0.30 / 0.05 =
#: 5.999999999999999) turning an exact tick into a spurious extra tick.
_TICK_EPS_DECIMALS = 9

#: Opposite side, used to flip a negatively-weighted component.
_OTHER: dict[str, str] = {"bid": "ask", "ask": "bid"}


# --------------------------------------------------------------------- #
# Spread convention
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpreadLegs:
    """How the spread instrument is composed from its two legs::

        spread = front_coef * front + back_coef * back

    For a 1:1 calendar spread the coefficients are +1 and -1 in some order.
    They are stored as floats rather than a bool so a ratio spread (e.g. a
    butterfly wing or a 3:2 crack) can reuse the same machinery.
    """

    front_coef: float
    back_coef: float

    def __post_init__(self) -> None:
        if self.front_coef == 0 or self.back_coef == 0:
            raise ValueError("both legs must carry a non-zero coefficient")
        if np.sign(self.front_coef) == np.sign(self.back_coef):
            raise ValueError(
                "a calendar spread is long one leg and short the other; got "
                f"front_coef={self.front_coef}, back_coef={self.back_coef}"
            )

    @property
    def name(self) -> str:
        return "back - front" if self.back_coef > 0 else "front - back"


#: Equity index (ES, NQ, ...): ESM6-ESU6 quotes +60.78 = deferred - nearby.
BACK_MINUS_FRONT = SpreadLegs(front_coef=-1.0, back_coef=1.0)

#: Grains (ZS, ZC, ZW) and most physicals: ZSF6-ZSH6 quotes -15.50 =
#: nearby - deferred.
FRONT_MINUS_BACK = SpreadLegs(front_coef=1.0, back_coef=-1.0)


def infer_spread_convention(
    front: pd.DataFrame,
    back: pd.DataFrame,
    spread: pd.DataFrame,
    tolerance: float | None = None,
) -> SpreadLegs:
    """Read the sign convention off the data instead of assuming it.

    Compares the spread's mid against ``back - front`` and ``front - back``
    over the aligned window and returns whichever matches. Cheap insurance:
    the two candidates differ by twice the spread level, so the comparison
    is never close, and getting it wrong is silent.

    ``tolerance`` defaults to a quarter of the spread's own median absolute
    level -- generous enough to absorb the fact that the legs and the spread
    are quoted on independent clocks, tight enough that the wrong sign can
    never pass.
    """
    aligned = align_books(
        front=top_of_book(front), back=top_of_book(back), spread=top_of_book(spread)
    )
    mid = lambda t: (t["bid_px"] + t["ask_px"]) / 2.0  # noqa: E731
    s = mid(aligned["spread"])
    diff = mid(aligned["back"]) - mid(aligned["front"])

    err_back_minus_front = float((s - diff).abs().median())
    err_front_minus_back = float((s + diff).abs().median())
    if tolerance is None:
        tolerance = max(float(s.abs().median()) / 4.0, 1e-9)

    if err_back_minus_front <= tolerance < err_front_minus_back:
        return BACK_MINUS_FRONT
    if err_front_minus_back <= tolerance < err_back_minus_front:
        return FRONT_MINUS_BACK
    raise ValueError(
        "cannot infer the spread convention: median |spread - (back-front)| = "
        f"{err_back_minus_front:.4f}, median |spread - (front-back)| = "
        f"{err_front_minus_back:.4f}, tolerance {tolerance:.4f}. Either the "
        "books are misaligned or this is not a 1:1 calendar spread -- pass "
        "SpreadLegs explicitly."
    )


# --------------------------------------------------------------------- #
# Book plumbing
# --------------------------------------------------------------------- #


def top_of_book(book: pd.DataFrame) -> pd.DataFrame:
    """Level-00 quote of one book, as ``bid_px/bid_sz/ask_px/ask_sz``.

    Rows where a side is empty keep their NaN; every downstream arithmetic
    op propagates it, so a one-sided book yields a one-sided implied quote
    rather than a fabricated one. Duplicate timestamps (several MBP events
    stamped identically) collapse to the last, which is the book state an
    as-of reader at that instant would see.
    """
    missing = [c for c in TOP_BOOK_COLUMNS if c not in book.columns]
    if missing:
        raise ValueError(f"book is missing top-of-book columns: {missing}")

    top = book.loc[:, list(TOP_BOOK_COLUMNS)].copy()
    top.columns = list(TOP_COLUMNS)
    top = top.astype(float)
    return top[~top.index.duplicated(keep="last")].sort_index()


def align_books(**books: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Put several top-of-book frames on one event-driven index.

    The index is the sorted union of every input's timestamps, forward
    filled, and truncated at the left to the point where *all* inputs have
    published at least once -- before that instant there is no implied
    quote, only a half-known one. Keyword names are preserved in the
    returned mapping so callers can unpack by leg name.

    Nothing here resamples. An implied quote is an event-driven object: it
    changes exactly when one of its components changes, and nowhere else.
    Snapping to a fixed grid (as ``resample_mids`` does for the correlation
    work) would both invent updates that never happened and drop ones that
    did.

    Raises if any book is empty: an implied quote with a missing component
    is not a wide quote, it is no quote, and silently returning an empty
    frame hides a bad fetch window.
    """
    if len(books) < 2:
        raise ValueError(f"need >=2 books to align, got {len(books)}")
    for name, frame in books.items():
        if frame.empty:
            raise ValueError(f"{name}: empty book")

    start = max(frame.index.min() for frame in books.values())
    index = pd.DatetimeIndex(
        sorted(set().union(*(frame.index for frame in books.values())))
    )
    index = index[index >= start]
    index.name = next(iter(books.values())).index.name

    return {
        name: frame.reindex(index, method="ffill") for name, frame in books.items()
    }


# --------------------------------------------------------------------- #
# The synthetic-quote rule
# --------------------------------------------------------------------- #


def synthetic_side(
    components: Sequence[tuple[float, pd.DataFrame]],
    side: Side,
) -> tuple[pd.Series, pd.Series]:
    """One side of a synthetic instrument ``X = sum_i c_i * Y_i``.

    Returns ``(price, size)``.

    Each component is read on the side you would actually have to trade to
    put the package on. Building the *bid* means selling X, so a component
    with ``c_i > 0`` is sold (read its ``bid``) and one with ``c_i < 0`` is
    bought (read its ``ask``). Building the *ask* flips both. That single
    sign test is the whole of implied quoting -- and it is what makes
    ``front_bid + spread_bid`` right for ES and wrong for soybeans.

    Size is the smallest component quantity, each scaled by its own
    coefficient and floored: one lot of X needs ``|c_i|`` lots of each
    ``Y_i``, all at once, so the package is capped by the thinnest
    component. This is why implied liquidity thins faster than any single
    input suggests -- a 4141-lot spread against a 23-lot outright implies
    23, not 4141.

    Note that *routes* aggregate where *legs* do not: two different
    combinations that imply the same price contribute additive size, so a
    caller comparing against a venue's displayed quote should sum across
    routes rather than take a min.
    """
    if len(components) < 2:
        raise ValueError(f"need >=2 components, got {len(components)}")
    if side not in ("bid", "ask"):
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

    index = components[0][1].index
    price = pd.Series(0.0, index=index)
    sizes = []
    for coef, quote in components:
        if not quote.index.equals(index):
            raise ValueError("components must share an index; align them first")
        # Positive coefficient -> trade this leg in the same direction as the
        # package; negative -> the opposite direction, so read the other side.
        leg_side = side if coef > 0 else _OTHER[side]
        price = price + coef * quote[f"{leg_side}_px"]
        sizes.append(np.floor(quote[f"{leg_side}_sz"] / abs(coef)))

    size = pd.concat(sizes, axis=1).min(axis=1, skipna=False)
    price.name, size.name = f"{side}_px", f"{side}_sz"
    return price, size


def round_to_tick(price, tick: float, direction: Direction):
    """Snap ``price`` onto the ``tick`` grid, always the conservative way.

    An implied price is arithmetic on other books' prices, and that
    arithmetic does not have to land on the tick grid of the instrument
    being quoted. ES is the live case: the outright trades in 0.25 but the
    calendar spread trades in 0.05, so ``front_ask + spread_ask`` is a
    multiple of 0.05 that is usually *not* a multiple of 0.25. It cannot be
    shown as an outright quote until it is moved to a tradeable price.

    Which way it moves is not a matter of taste. Rounding an offer *down*
    or a bid *up* would advertise a price better than the legs can actually
    produce -- the quote would be unhedgeable, and lifting it would be a
    guaranteed loss of up to one tick. So offers round ``"up"`` and bids
    round ``"down"``: the implied quote is never better than its synthetic
    cost, and the leftover fraction of a tick is edge, not risk.

    Accepts scalars or Series; NaN in, NaN out.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if not tick > 0:
        raise ValueError(f"tick must be positive, got {tick!r}")

    ratio = np.round(np.asarray(price, dtype=float) / tick, _TICK_EPS_DECIMALS)
    snapped = (np.ceil(ratio) if direction == "up" else np.floor(ratio)) * tick
    # ``n * tick`` reintroduces float dust (6 * 0.05 -> 0.30000000000000004),
    # which then leaks into the round-edge columns and into any equality
    # check a caller makes against a quoted price. Snap it back off.
    snapped = np.round(snapped, _TICK_EPS_DECIMALS)

    if isinstance(price, pd.Series):
        return pd.Series(snapped, index=price.index, name=price.name)
    return snapped


#: Column order produced by ``assemble_quote``; both modules re-export it.
QUOTE_COLUMNS = (
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
    "raw_bid_px",
    "raw_ask_px",
    "bid_round_edge",
    "ask_round_edge",
    "width",
)


def combine_routes(routes: Sequence[pd.DataFrame], tol: float = 1e-9) -> pd.DataFrame:
    """Reduce several implied quotes for the same instrument to the best one.

    Each element of ``routes`` is a frame of ``QUOTE_COLUMNS`` produced by a
    different combination that implies the same instrument. The best bid is
    the highest across routes; the best ask the lowest.

    **Sizes add across routes, where they min within a route.** A route is a
    package that needs every one of its components at once, so it is capped
    by its thinnest -- but two independent routes at the same price are two
    independent parcels of liquidity, and a venue displays their sum. In the
    soybean snapshot both routes to a 1026.25 ZSF6 bid are capped by their
    outright leg (23 via ZSX5, 16 via ZSH6), so the implied bid shows 39,
    on top of the 72 resting natively.

    Sizes are summed only over the routes actually *at* the best price;
    routes behind it contribute nothing, exactly as they would in a book.
    """
    if not routes:
        raise ValueError("need at least one route")
    index = routes[0].index
    for r in routes:
        if not r.index.equals(index):
            raise ValueError("routes must share an index; align them first")

    # Stack as plain arrays: every route carries identically-named columns,
    # so concatenating the Series directly would collide on the column axis.
    def _stack(col) -> np.ndarray:
        return np.column_stack([r[col].to_numpy(dtype=float) for r in routes])

    bid_px, ask_px = _stack("bid_px"), _stack("ask_px")
    best_bid = pd.Series(np.nanmax(bid_px, axis=1), index=index)
    best_ask = pd.Series(np.nanmin(ask_px, axis=1), index=index)

    def _size_at(best: pd.Series, prices: np.ndarray, sizes: np.ndarray) -> pd.Series:
        at_best = np.abs(prices - best.to_numpy()[:, None]) <= tol
        totals = np.where(at_best, np.nan_to_num(sizes), 0.0).sum(axis=1)
        # No route at the best price means there is no quote, not a zero.
        return pd.Series(np.where(at_best.any(axis=1), totals, np.nan), index=index)

    return pd.DataFrame(
        {
            "bid_px": best_bid,
            "bid_sz": _size_at(best_bid, bid_px, _stack("bid_sz")),
            "ask_px": best_ask,
            "ask_sz": _size_at(best_ask, ask_px, _stack("ask_sz")),
            "raw_bid_px": best_bid,
            "raw_ask_px": best_ask,
            "bid_round_edge": 0.0,
            "ask_round_edge": 0.0,
            "width": best_ask - best_bid,
        },
        index=index,
    ).loc[:, list(QUOTE_COLUMNS)]


def assemble_quote(
    components: Sequence[tuple[float, pd.DataFrame]],
    tick: float,
) -> pd.DataFrame:
    """Both sides of a synthetic instrument, rounded onto ``tick``.

    Shared tail of both public modules. Returns ``QUOTE_COLUMNS``: the
    tradeable ``bid_px``/``ask_px`` and their sizes, the unrounded
    ``raw_*_px`` (the true economics of the package before it is made
    showable), the ``*_round_edge`` conceded to the tick grid (always >= 0),
    and the resulting ``width``.
    """
    raw_bid, bid_sz = synthetic_side(components, "bid")
    raw_ask, ask_sz = synthetic_side(components, "ask")

    bid_px = round_to_tick(raw_bid, tick, "down")
    ask_px = round_to_tick(raw_ask, tick, "up")

    out = pd.DataFrame(
        {
            "bid_px": bid_px,
            "bid_sz": bid_sz,
            "ask_px": ask_px,
            "ask_sz": ask_sz,
            "raw_bid_px": raw_bid,
            "raw_ask_px": raw_ask,
            # Bids round down and offers round up, so both differences are
            # signed to be non-negative: edge conceded, never claimed.
            "bid_round_edge": raw_bid - bid_px,
            "ask_round_edge": ask_px - raw_ask,
            "width": ask_px - bid_px,
        },
        index=components[0][1].index,
    )
    return out.loc[:, list(QUOTE_COLUMNS)]
