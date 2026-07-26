"""Empirical lead-lag & co-movement analysis of front / back / spread (Issue #11).

Pure functions over ``CalendarSpreadData`` frames — no fetching, no disk I/O —
so every piece is unit-testable on synthetic books. The driver
(``run_relationship_analysis.py``) wires this to the cached fetcher.

This module produces the Goal-2 empirical footing that Issue #14's quoting
strategy assumes and Issue #15 later formalizes:

- the front month is the most liquid book and leads price discovery;
- the calendar spread leads the basis (the outright term structure);
- the deferred leg follows both, its level being ~ ``front + spread``.

Method notes (why the code is shaped this way)
-----------------------------------------------
1. Alignment. The three books carry their own raw event timestamps and are
   NOT on a common clock (see ``CalendarSpreadData``). Anything cross-
   instrument therefore starts by sampling every book's top-of-book mid onto
   one regular grid with as-of / last-value-carried-forward semantics
   (``resample_mids``). This is the crux; get it wrong and every correlation
   below is meaningless.
2. Levels vs changes. Lead-lag and correlation run on first differences —
   price *levels* are non-stationary (and cointegrated across the three
   instruments), which manufactures spurious ~1.0 correlations. The one place
   levels are used on purpose is the structural ``deferred ~ front + spread``
   identity in ``deferred_vs_implied`` (a cointegration relationship).
3. Mids, not trade prints. Mids avoid the bid-ask bounce that would inject
   noise into a co-movement study.
4. Sampling interval. Too fine and cross-correlations attenuate from
   asynchronous quoting (the Epps effect); too coarse and the lead-lag signal
   smears out. ``freq`` is a parameter so the driver can sweep it and report
   the sensitivity rather than hard-coding one answer.

Dependencies are numpy + pandas only (both declared in pyproject); no scipy /
statsmodels, so the OLS here is a plain ``numpy.linalg.lstsq``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from market_data_fetcher import CalendarSpreadContractSpec, CalendarSpreadData

#: The hypothesis pairs from the README, as (leader_candidate, follower).
#: ``basis`` and ``implied_deferred`` are derived columns added by
#: ``resample_mids``.
DEFAULT_LEAD_LAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("front_mid", "back_mid"),      # expect the front to lead
    ("spread_mid", "basis"),        # expect the spread to lead the basis
    ("front_mid", "spread_mid"),    # reported for context
)


# --------------------------------------------------------------------------- #
# Mids and the aligned panel
# --------------------------------------------------------------------------- #
def top_mid(book: pd.DataFrame) -> pd.Series:
    """Top-of-book mid ``(bid_px_00 + ask_px_00) / 2`` for one book frame.

    Returns a Series indexed by the book's ``ts_event`` index (name ``mid``).
    Rows with an empty top level (NaN price) yield NaN mids.
    """
    mid = (book["bid_px_00"] + book["ask_px_00"]) / 2.0
    mid.name = "mid"
    return mid


def _dedup_last(s: pd.Series) -> pd.Series:
    """Keep the last observation at any repeated timestamp (current state)."""
    return s[~s.index.duplicated(keep="last")]


def resample_mids(data: CalendarSpreadData, freq: str = "500ms") -> pd.DataFrame:
    """Front/back/spread mids on ONE common grid, plus derived columns.

    Each book's top-of-book mid is sampled onto a regular ``freq`` grid with
    last-value-carried-forward (as-of) semantics. The grid starts at the
    latest of the three first-event times, so every column has a real state at
    every grid point (no leading NaNs from a book that had not started yet).

    Columns:
        front_mid, back_mid, spread_mid : sampled mids
        basis            = back_mid - front_mid   (outright term structure)
        implied_deferred = front_mid + spread_mid (what #14 will quote from)

    ``implied_deferred`` should track ``back_mid`` closely and ``basis`` should
    track ``spread_mid`` closely — that is the co-movement Goal 2 rests on.

    Raises:
        ValueError: if any of the three books is empty.
    """
    mids = {
        "front_mid": _dedup_last(top_mid(data.front)),
        "back_mid": _dedup_last(top_mid(data.back)),
        "spread_mid": _dedup_last(top_mid(data.spread)),
    }
    for name, s in mids.items():
        if s.empty:
            raise ValueError(f"{name}: book is empty, cannot build a mid panel")

    start = max(s.index.min() for s in mids.values())
    end = max(s.index.max() for s in mids.values())
    grid = pd.date_range(start=start, end=end, freq=freq)
    if len(grid) == 0:  # start == end or sub-freq window
        grid = pd.DatetimeIndex([start])
    grid.name = data.front.index.name  # ts_event

    panel = pd.DataFrame(
        {name: s.reindex(grid, method="ffill") for name, s in mids.items()}
    )
    panel["basis"] = panel["back_mid"] - panel["front_mid"]
    panel["implied_deferred"] = panel["front_mid"] + panel["spread_mid"]
    return panel


# --------------------------------------------------------------------------- #
# Liquidity characterization
# --------------------------------------------------------------------------- #
def _instrument_liquidity(
    book: pd.DataFrame,
    trades: pd.DataFrame,
    tick_size: float,
    multiplier: float,
) -> dict:
    """Liquidity metrics for a single instrument's book + trade frames."""
    quoted_spread_pts = (book["ask_px_00"] - book["bid_px_00"]).mean()
    top_depth = ((book["bid_sz_00"] + book["ask_sz_00"]) / 2.0).mean()

    if len(book) > 1:
        duration_s = (book.index[-1] - book.index[0]).total_seconds()
    else:
        duration_s = 0.0
    update_rate = len(book) / duration_s if duration_s > 0 else float("nan")

    return {
        "n_updates": float(len(book)),
        "update_rate_per_s": update_rate,
        "avg_spread_pts": float(quoted_spread_pts),
        "avg_spread_ticks": float(quoted_spread_pts / tick_size),
        "avg_spread_usd": float(quoted_spread_pts * multiplier),
        "avg_top_depth": float(top_depth),
        "n_trades": float(len(trades)),
        "trade_volume": float(trades["size"].sum()) if len(trades) else 0.0,
        "avg_trade_size": float(trades["size"].mean()) if len(trades) else float("nan"),
    }


def liquidity_metrics(
    data: CalendarSpreadData, spec: CalendarSpreadContractSpec
) -> pd.DataFrame:
    """One row per instrument (front, back, spread) of liquidity measures.

    Establishes the liquidity ranking (expected: front > spread > deferred).
    Front and back use ``spec.outright_tick_size``; the spread uses
    ``spec.spread_tick_size``. Dollar figures use ``spec.contract_multiplier``.
    """
    mult = float(spec.contract_multiplier)
    outright_tick = float(spec.outright_tick_size)
    spread_tick = float(spec.spread_tick_size)

    rows = {
        "front": _instrument_liquidity(data.front, data.front_trades, outright_tick, mult),
        "back": _instrument_liquidity(data.back, data.back_trades, outright_tick, mult),
        "spread": _instrument_liquidity(data.spread, data.spread_trades, spread_tick, mult),
    }
    out = pd.DataFrame(rows).T
    out.index.name = "instrument"
    return out


# --------------------------------------------------------------------------- #
# Co-movement
# --------------------------------------------------------------------------- #
def _changes(panel: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """First differences of the given columns (default: the mid + basis set)."""
    if columns is None:
        columns = [c for c in ("front_mid", "back_mid", "spread_mid", "basis") if c in panel]
    return panel[list(columns)].diff().dropna(how="all")


def change_corr(panel: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Contemporaneous correlation matrix of first differences (not levels).

    A column that never moves in the window (zero variance — common for the
    spread mid on coarse grids) yields NaN correlations rather than a numpy
    divide warning.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return _changes(panel, columns).corr()


def deferred_vs_implied(panel: pd.DataFrame) -> dict:
    """OLS ``back_mid ~ 1 + front_mid + spread_mid`` (a levels/cointegration fit).

    R² near 1 with slopes near (1, 1) and a small intercept is the empirical
    green light for #14's "imply the deferred leg from front + spread" idea.

    Returns intercept, ``beta_front``, ``beta_spread``, ``r_squared``,
    ``resid_std`` (points) and the row count ``n``.
    """
    df = panel[["front_mid", "spread_mid", "back_mid"]].dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"need >=3 aligned rows for the regression, got {n}")

    X = np.column_stack([np.ones(n), df["front_mid"].to_numpy(), df["spread_mid"].to_numpy()])
    y = df["back_mid"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "intercept": float(coef[0]),
        "beta_front": float(coef[1]),
        "beta_spread": float(coef[2]),
        "r_squared": float(r2),
        "resid_std": float(resid.std(ddof=len(coef))) if n > len(coef) else float("nan"),
        "n": float(n),
    }


# --------------------------------------------------------------------------- #
# Lead-lag
# --------------------------------------------------------------------------- #
def cross_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> pd.Series:
    """Cross-correlation ``corr(a_t, b_{t+k})`` for ``k`` in ``[-max_lag, max_lag]``.

    ``a`` and ``b`` should be *change* series on the same index. A positive
    peak lag means ``a`` leads ``b`` (today's ``a`` correlates with a future
    ``b``). Correlations are Pearson, computed pairwise-complete per lag.

    Returns a Series indexed by integer lag ``k`` (name ``cross_corr``).
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    lags = range(-max_lag, max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = {k: a.corr(b.shift(-k)) for k in lags}
    out = pd.Series(vals, name="cross_corr")
    out.index.name = "lag"
    return out


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int) -> dict:
    """Peak-lag summary of ``cross_correlation``.

    Picks the lag of maximum absolute correlation (so a strong negative
    relationship is not missed) and reports the signed correlation there.

    Returns ``peak_lag`` (int), ``peak_corr`` (signed float) and ``leads``:
    ``"a"`` if ``a`` leads (peak_lag > 0), ``"b"`` if ``b`` leads
    (peak_lag < 0), ``"contemporaneous"`` at 0.
    """
    ccf = cross_correlation(a, b, max_lag)
    if ccf.abs().isna().all():
        return {"peak_lag": 0, "peak_corr": float("nan"), "leads": "undetermined"}
    peak_lag = int(ccf.abs().idxmax())
    peak_corr = float(ccf.loc[peak_lag])
    leads = "a" if peak_lag > 0 else "b" if peak_lag < 0 else "contemporaneous"
    return {"peak_lag": peak_lag, "peak_corr": peak_corr, "leads": leads}


def lead_lag_table(
    panel: pd.DataFrame,
    max_lag: int = 20,
    pairs: Iterable[tuple[str, str]] = DEFAULT_LEAD_LAG_PAIRS,
) -> pd.DataFrame:
    """Run ``lead_lag`` on each hypothesis pair (on first differences).

    One row per pair: the two columns, the peak lag and correlation, and a
    human-readable interpretation of who leads.
    """
    rows = []
    for a_col, b_col in pairs:
        changes = panel[[a_col, b_col]].diff().dropna()
        res = lead_lag(changes[a_col], changes[b_col], max_lag)
        if res["leads"] == "a":
            interp = f"{a_col} leads {b_col} by {res['peak_lag']} step(s)"
        elif res["leads"] == "b":
            interp = f"{b_col} leads {a_col} by {abs(res['peak_lag'])} step(s)"
        elif res["leads"] == "contemporaneous":
            interp = f"{a_col} and {b_col} move contemporaneously"
        else:
            interp = "undetermined (no variation)"
        rows.append(
            {
                "a": a_col,
                "b": b_col,
                "peak_lag": res["peak_lag"],
                "peak_corr": res["peak_corr"],
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Direct spread-vs-basis test (Issue #11)
# --------------------------------------------------------------------------- #
def spread_vs_basis(panel: pd.DataFrame) -> dict:
    """Test the "spread ~ basis" identity directly, without the front dominating.

    The full ``back ~ front + spread`` regression identifies the front slope
    well but leaves the spread slope weakly identified whenever the spread
    barely moves in the window (near-zero variance -> unstable coefficient,
    even at R^2 ~ 1). This isolates the specific Goal-2 claim that the
    calendar spread IS the outright basis by regressing

        basis (= back_mid - front_mid)  ~  1 + spread_mid

    and also reports the correlation of their *changes* (the level fit can look
    strong purely from cointegration; the change correlation is the honest
    co-movement check). ``spread_activity`` is carried through as the caveat
    flag: a small value means the spread was too static to trust either number.

    Returns intercept, ``beta_spread``, ``r_squared`` (levels),
    ``change_corr`` (first differences), ``resid_std``, ``n`` and
    ``spread_n_changes`` (how many grid steps the spread actually moved).
    """
    df = panel[["spread_mid", "basis"]].dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"need >=3 aligned rows, got {n}")

    x = df["spread_mid"].to_numpy()
    y = df["basis"].to_numpy()
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    d = df.diff().dropna()
    spread_n_changes = int((d["spread_mid"] != 0).sum())
    if spread_n_changes > 0 and len(d) > 1:
        change_corr_val = float(d["spread_mid"].corr(d["basis"]))
    else:
        change_corr_val = float("nan")  # spread never moved: correlation undefined

    return {
        "intercept": float(coef[0]),
        "beta_spread": float(coef[1]),
        "r_squared": float(r2),
        "change_corr": change_corr_val,
        "resid_std": float(resid.std(ddof=2)) if n > 2 else float("nan"),
        "n": float(n),
        "spread_n_changes": float(spread_n_changes),
    }


# --------------------------------------------------------------------------- #
# Window liveliness (Issue #11) — is this window worth analyzing?
# --------------------------------------------------------------------------- #
def spread_activity(data: CalendarSpreadData) -> dict:
    """How much the spread instrument actually moved/traded in this window.

    The lead-lag and spread-vs-basis tests are only meaningful when the spread
    has variation to detect. This summarizes it so the caller (or
    ``liveliest_subwindow``) can decide whether a window is worth trusting or
    should be swapped for one nearer the roll.

    Returns spread book-update count, distinct mid moves, trade count and
    volume, and the window's start/end.
    """
    spread = data.spread
    mid = _dedup_last(top_mid(spread))
    n_moves = int((mid.diff().fillna(0) != 0).sum())
    return {
        "spread_updates": int(len(spread)),
        "spread_mid_moves": n_moves,
        "spread_trades": int(len(data.spread_trades)),
        "spread_volume": float(data.spread_trades["size"].sum()) if len(data.spread_trades) else 0.0,
        "start": spread.index.min() if len(spread) else None,
        "end": spread.index.max() if len(spread) else None,
    }


def liveliest_subwindow(
    data: CalendarSpreadData,
    n_bins: int = 6,
    metric: str = "spread_trades",
) -> dict:
    """Find which equal slice of the window has the most spread activity.

    Splits the window into ``n_bins`` equal time slices and ranks them by
    ``metric`` ("spread_trades" or "spread_updates"). Use it to point a closer
    analysis at the busiest stretch of an otherwise-quiet window, or — when
    even the best slice is nearly empty — as evidence that the whole window is
    too static and you should fetch one nearer the quarterly roll.

    Returns the best slice's index, its [start, end], its activity counts, and
    a per-bin table (list of dicts) so the caller can see the distribution.
    """
    spread = data.spread
    trades = data.spread_trades
    if len(spread) == 0:
        raise ValueError("spread book is empty")

    start = spread.index.min()
    end = spread.index.max()
    edges = pd.date_range(start=start, end=end, periods=n_bins + 1)

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include the right edge only in the last bin
        upd = spread.loc[(spread.index >= lo) & (spread.index < hi)]
        trd = trades.loc[(trades.index >= lo) & (trades.index < hi)] if len(trades) else trades
        if i == n_bins - 1:
            upd = spread.loc[(spread.index >= lo) & (spread.index <= hi)]
            trd = trades.loc[(trades.index >= lo) & (trades.index <= hi)] if len(trades) else trades
        bins.append(
            {
                "bin": i,
                "start": lo,
                "end": hi,
                "spread_updates": int(len(upd)),
                "spread_trades": int(len(trd)),
            }
        )

    if metric not in ("spread_trades", "spread_updates"):
        raise ValueError(f"metric must be spread_trades or spread_updates, got {metric!r}")
    best = max(bins, key=lambda b: b[metric])
    return {
        "best_bin": best["bin"],
        "start": best["start"],
        "end": best["end"],
        "spread_updates": best["spread_updates"],
        "spread_trades": best["spread_trades"],
        "n_bins": n_bins,
        "metric": metric,
        "bins": bins,
    }


# --------------------------------------------------------------------------- #
# Spread mid "stickiness" diagnostic (Issue #11)
# --------------------------------------------------------------------------- #
def spread_mid_diagnostic(data: CalendarSpreadData, spec: CalendarSpreadContractSpec) -> dict:
    """How much the spread MID actually varied — the "sticky mid" check.

    Heavy spread trading with a near-constant mid is why the spread-vs-basis
    and lead-lag tests stay weak even in busy roll windows: hundreds of trades
    can print at essentially one price, leaving no mid variation to correlate.
    This quantifies that directly on the raw (un-resampled) spread book so the
    writeup can state the fact rather than infer it.

    Returns:
        n_book_rows       : spread book updates in the window
        n_distinct_mids   : how many distinct mid prices the spread took
        mid_range_ticks   : (max - min) mid, in spread ticks
        n_mid_moves       : number of times the mid changed
        move_size_ticks   : {mean, max} absolute mid move, in spread ticks,
                            over the steps where it moved (empty -> nan)
        distinct_mids     : the sorted distinct mid values (handy for a
                            "the mid took only these N values" sentence)
    """
    tick = float(spec.spread_tick_size)
    mid = _dedup_last(top_mid(data.spread)).dropna()
    if mid.empty:
        raise ValueError("spread book has no usable top-of-book mids")

    distinct = sorted(mid.unique().tolist())
    diffs = mid.diff().dropna()
    moves = diffs[diffs != 0].abs()  # points
    return {
        "n_book_rows": int(len(data.spread)),
        "n_distinct_mids": int(len(distinct)),
        "mid_range_ticks": float((mid.max() - mid.min()) / tick),
        "n_mid_moves": int(len(moves)),
        "move_size_ticks": {
            "mean": float((moves.mean()) / tick) if len(moves) else float("nan"),
            "max": float((moves.max()) / tick) if len(moves) else float("nan"),
        },
        "distinct_mids": distinct,
    }


# --------------------------------------------------------------------------- #
# Spread-vs-basis frequency sweep (Issue #11)
# --------------------------------------------------------------------------- #
def spread_basis_sweep(
    data: CalendarSpreadData,
    freqs: Iterable[str] = ("500ms", "1s", "2s", "5s", "10s", "30s"),
    max_lag_seconds: float = 120.0,
) -> pd.DataFrame:
    """How the spread<->basis relationship changes with sampling interval.

    Motivation: ``basis = back - front`` is a small difference of two fast,
    independently-noisy leg books, so at fine grids its changes are dominated
    by leg micro-noise and the spread<->basis correlation is ~0. Averaging over
    coarser intervals cancels that independent noise while any real co-movement
    persists, so the correlation should rise with the interval. This sweep makes
    that curve explicit.

    For each ``freq`` it reports, on first differences of spread_mid and basis:
      - ``zerolag_corr``  : contemporaneous change-correlation
      - ``peak_lag_s``    : lag (seconds) of maximum |cross-correlation|;
                            >0 means spread LEADS basis, <0 means basis LEADS
                            spread (a=spread, b=basis)
      - ``peak_corr``     : signed correlation at that lag

    ``max_lag_seconds`` bounds the lead-lag scan so it is comparable across
    grids; per grid it is also capped so it never exceeds ~1/3 of the sample.
    """
    rows = []
    for freq in freqs:
        panel = resample_mids(data, freq=freq)
        d = panel[["spread_mid", "basis"]].diff().dropna()
        n = len(d)

        moves = int((d["spread_mid"] != 0).sum())
        if n > 2 and d["spread_mid"].std() > 0 and d["basis"].std() > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                zerolag = float(d["spread_mid"].corr(d["basis"]))
        else:
            zerolag = float("nan")

        step_s = pd.Timedelta(freq).total_seconds()
        max_lag_steps = max(1, int(round(max_lag_seconds / step_s)))
        max_lag_steps = min(max_lag_steps, max(1, n // 3))  # keep lags vs sample sane
        ccf = cross_correlation(d["spread_mid"], d["basis"], max_lag_steps)
        if ccf.abs().isna().all():
            peak_lag_s, peak_corr = float("nan"), float("nan")
        else:
            peak_step = int(ccf.abs().idxmax())
            peak_lag_s = peak_step * step_s
            peak_corr = float(ccf.loc[peak_step])

        rows.append(
            {
                "freq": freq,
                "grid_points": len(panel),
                "spread_moves": moves,
                "zerolag_corr": round(zerolag, 4),
                "peak_lag_s": peak_lag_s,
                "peak_corr": round(peak_corr, 4),
            }
        )
    return pd.DataFrame(rows)
