"""Lead-lag and co-movement of the front/back/spread books (issue #11).

Pure functions over CalendarSpreadData; run_relationship_analysis.py wires in
the fetcher. The three books have independent timestamps, so everything starts
from resample_mids (mids on a common grid). Correlations and lead-lag run on
first differences, not price levels. numpy + pandas only.

Implied matching ties the three quotes mechanically, so a lead-lag on the mids
peaks at lag 0 by construction -- use trade_lead_lag_table for genuine
price-discovery lag, and spread_basis_sweep for the spread/basis co-movement.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from market_data_fetcher import CalendarSpreadContractSpec, CalendarSpreadData

# (leader candidate, follower) pairs we test. basis/implied_deferred are added
# by resample_mids.
DEFAULT_LEAD_LAG_PAIRS = (
    ("front_mid", "back_mid"),
    ("spread_mid", "basis"),
    ("front_mid", "spread_mid"),
)


def top_mid(book: pd.DataFrame) -> pd.Series:
    """Simple (bid + ask) / 2. Bounces a tick in one-tick books; used for the
    sticky-mid diagnostics, not the correlation panel (see micro_price)."""
    mid = (book["bid_px_00"] + book["ask_px_00"]) / 2.0
    mid.name = "mid"
    return mid


def micro_price(book: pd.DataFrame) -> pd.Series:
    """Size-weighted mid (bid_px*ask_sz + ask_px*bid_sz)/(bid_sz+ask_sz).

    Leans toward the thinner side instead of bouncing a full tick when the touch
    flips. Falls back to the simple mid on an empty touch.
    """
    bid_px, ask_px = book["bid_px_00"], book["ask_px_00"]
    bid_sz, ask_sz = book["bid_sz_00"], book["ask_sz_00"]
    total = bid_sz + ask_sz
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = (bid_px * ask_sz + ask_px * bid_sz) / total
    simple = (bid_px + ask_px) / 2.0
    out = micro.where(total > 0, simple)
    out.name = "mid"
    return out


def _dedup_last(s: pd.Series) -> pd.Series:
    """Keep the last value at each timestamp."""
    return s[~s.index.duplicated(keep="last")]


def resample_mids(
    data: CalendarSpreadData, freq: str = "500ms", price_fn=micro_price
) -> pd.DataFrame:
    """Front/back/spread mids on one ffilled grid, plus basis (back - front)
    and implied_deferred (front + spread).

    micro_price by default; pass price_fn=top_mid for the simple mid. Grid
    starts where all three books have data. Raises if any book is empty.
    """
    mids = {
        "front_mid": _dedup_last(price_fn(data.front)),
        "back_mid": _dedup_last(price_fn(data.back)),
        "spread_mid": _dedup_last(price_fn(data.spread)),
    }
    for name, s in mids.items():
        if s.empty:
            raise ValueError(f"{name}: empty book")

    start = max(s.index.min() for s in mids.values())
    end = max(s.index.max() for s in mids.values())
    grid = pd.date_range(start=start, end=end, freq=freq)
    if len(grid) == 0:
        grid = pd.DatetimeIndex([start])
    grid.name = data.front.index.name

    panel = pd.DataFrame({name: s.reindex(grid, method="ffill") for name, s in mids.items()})
    panel["basis"] = panel["back_mid"] - panel["front_mid"]
    panel["implied_deferred"] = panel["front_mid"] + panel["spread_mid"]
    return panel


def _instrument_liquidity(book, trades, tick_size, multiplier) -> dict:
    quoted_spread = (book["ask_px_00"] - book["bid_px_00"]).mean()
    depth = ((book["bid_sz_00"] + book["ask_sz_00"]) / 2.0).mean()
    duration = (book.index[-1] - book.index[0]).total_seconds() if len(book) > 1 else 0.0
    return {
        "n_updates": float(len(book)),
        "update_rate_per_s": len(book) / duration if duration > 0 else float("nan"),
        "avg_spread_pts": float(quoted_spread),
        "avg_spread_ticks": float(quoted_spread / tick_size),
        "avg_spread_usd": float(quoted_spread * multiplier),
        "avg_top_depth": float(depth),
        "n_trades": float(len(trades)),
        "trade_volume": float(trades["size"].sum()) if len(trades) else 0.0,
        "avg_trade_size": float(trades["size"].mean()) if len(trades) else float("nan"),
    }


def liquidity_metrics(data: CalendarSpreadData, spec: CalendarSpreadContractSpec) -> pd.DataFrame:
    """Per-instrument liquidity. Front/back use the outright tick, spread the
    spread tick; dollar figures use the multiplier."""
    mult = float(spec.contract_multiplier)
    outright = float(spec.outright_tick_size)
    spread = float(spec.spread_tick_size)
    rows = {
        "front": _instrument_liquidity(data.front, data.front_trades, outright, mult),
        "back": _instrument_liquidity(data.back, data.back_trades, outright, mult),
        "spread": _instrument_liquidity(data.spread, data.spread_trades, spread, mult),
    }
    out = pd.DataFrame(rows).T
    out.index.name = "instrument"
    return out


def change_corr(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the mid/basis changes. A flat column gives NaN,
    not a divide warning."""
    cols = [c for c in ("front_mid", "back_mid", "spread_mid", "basis") if c in panel]
    with np.errstate(invalid="ignore", divide="ignore"):
        return panel[cols].diff().dropna(how="all").corr()


def deferred_vs_implied(panel: pd.DataFrame) -> dict:
    """OLS back_mid ~ front_mid + spread_mid (levels). Coarse mid diagnostic:
    R^2 ~ 1 is mechanical (implied matching), so read the slopes and resid_std,
    not the R^2. Tradeable version: implied_outright.implied_back_book."""
    df = panel[["front_mid", "spread_mid", "back_mid"]].dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"need >=3 rows, got {n}")

    X = np.column_stack([np.ones(n), df["front_mid"], df["spread_mid"]])
    y = df["back_mid"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept": float(coef[0]),
        "beta_front": float(coef[1]),
        "beta_spread": float(coef[2]),
        "r_squared": r2,
        "resid_std": float(resid.std(ddof=3)) if n > 3 else float("nan"),
        "n": float(n),
    }


def cross_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> pd.Series:
    """corr(a_t, b_{t+k}) for k in [-max_lag, max_lag]; +k means a leads b.
    Pass first-difference series."""
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = {k: a.corr(b.shift(-k)) for k in range(-max_lag, max_lag + 1)}
    out = pd.Series(vals, name="cross_corr")
    out.index.name = "lag"
    return out


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int) -> dict:
    """Lag of max |corr| from cross_correlation, with the signed corr and who
    leads (a if lag > 0, b if < 0)."""
    ccf = cross_correlation(a, b, max_lag)
    if ccf.abs().isna().all():
        return {"peak_lag": 0, "peak_corr": float("nan"), "leads": "undetermined"}
    peak_lag = int(ccf.abs().idxmax())
    leads = "a" if peak_lag > 0 else "b" if peak_lag < 0 else "contemporaneous"
    return {"peak_lag": peak_lag, "peak_corr": float(ccf.loc[peak_lag]), "leads": leads}


def _interpret(a_col: str, b_col: str, res: dict) -> str:
    """Human-readable lead-lag verdict for one (a, b) result."""
    if res["leads"] == "a":
        return f"{a_col} leads {b_col} by {res['peak_lag']}"
    if res["leads"] == "b":
        return f"{b_col} leads {a_col} by {abs(res['peak_lag'])}"
    if res["leads"] == "contemporaneous":
        return f"{a_col} and {b_col} contemporaneous"
    return "undetermined"


def lead_lag_table(panel, max_lag=20, pairs=DEFAULT_LEAD_LAG_PAIRS) -> pd.DataFrame:
    """lead_lag per (a, b) pair on first differences of the quote mids. On the
    mids this is the mechanical tie (expect lag 0); use trade_lead_lag_table
    for flow."""
    rows = []
    for a_col, b_col in pairs:
        ch = panel[[a_col, b_col]].diff().dropna()
        res = lead_lag(ch[a_col], ch[b_col], max_lag)
        rows.append({"a": a_col, "b": b_col, "peak_lag": res["peak_lag"],
                     "peak_corr": res["peak_corr"],
                     "interpretation": _interpret(a_col, b_col, res)})
    return pd.DataFrame(rows)


def spread_activity(data: CalendarSpreadData) -> dict:
    """How much the spread moved and traded in the window."""
    mid = _dedup_last(top_mid(data.spread))
    trades = data.spread_trades
    return {
        "spread_updates": int(len(data.spread)),
        "spread_mid_moves": int((mid.diff().fillna(0) != 0).sum()),
        "spread_trades": int(len(trades)),
        "spread_volume": float(trades["size"].sum()) if len(trades) else 0.0,
    }


def spread_mid_diagnostic(data: CalendarSpreadData, spec: CalendarSpreadContractSpec) -> dict:
    """Spread-mid variation: distinct mids, range in ticks, number of moves,
    mean move size. Explains weak spread tests when the mid is sticky."""
    tick = float(spec.spread_tick_size)
    mid = _dedup_last(top_mid(data.spread)).dropna()
    if mid.empty:
        raise ValueError("spread book has no mids")
    moves = mid.diff().dropna()
    moves = moves[moves != 0].abs()
    return {
        "n_distinct_mids": int(mid.nunique()),
        "mid_range_ticks": float((mid.max() - mid.min()) / tick),
        "n_mid_moves": int(len(moves)),
        "mean_move_ticks": float(moves.mean() / tick) if len(moves) else float("nan"),
    }


def spread_basis_sweep(
    data: CalendarSpreadData,
    freqs: Iterable[str] = ("500ms", "1s", "2s", "5s", "10s", "30s"),
    max_lag_seconds: float = 120.0,
) -> pd.DataFrame:
    """Spread vs basis across sampling intervals.

    basis = back - front is a noisy difference of two fast books, so at fine
    grids the spread<->basis correlation is ~0; coarser grids cancel the leg
    noise and it rises. peak_lag_s > 0 means spread leads basis, < 0 basis leads.
    """
    rows = []
    for freq in freqs:
        panel = resample_mids(data, freq=freq)
        d = panel[["spread_mid", "basis"]].diff().dropna()
        n = len(d)

        if n > 2 and d["spread_mid"].std() > 0 and d["basis"].std() > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                zerolag = float(d["spread_mid"].corr(d["basis"]))
        else:
            zerolag = float("nan")

        step_s = pd.Timedelta(freq).total_seconds()
        max_lag = min(max(1, round(max_lag_seconds / step_s)), max(1, n // 3))
        ccf = cross_correlation(d["spread_mid"], d["basis"], max_lag)
        if ccf.abs().isna().all():
            peak_lag_s, peak_corr = float("nan"), float("nan")
        else:
            peak_step = int(ccf.abs().idxmax())
            peak_lag_s, peak_corr = peak_step * step_s, float(ccf.loc[peak_step])

        rows.append({"freq": freq, "grid_points": len(panel),
                     "spread_moves": int((d["spread_mid"] != 0).sum()),
                     "zerolag_corr": round(zerolag, 4),
                     "peak_lag_s": peak_lag_s, "peak_corr": round(peak_corr, 4)})
    return pd.DataFrame(rows)


# Trade-based lead-lag
DEFAULT_TRADE_PAIRS = (
    ("front_flow", "back_flow"),
    ("front_flow", "spread_flow"),
)


def _signed_size(trades: pd.DataFrame) -> pd.Series:
    """+size for 'B', -size for 'A', 0 for 'N'/unknown."""
    sign = trades["side"].map({"B": 1.0, "A": -1.0}).fillna(0.0)
    return sign * trades["size"].astype(float)


def drop_implied_coincident(
    trades: pd.DataFrame, other: pd.DataFrame, tol: str = "2ms"
) -> pd.DataFrame:
    """Drop trades within +/- tol of any trade in ``other``. The schema has no
    implied-match flag, so implied fills are proxied by their signature -- a
    print coincident with a spread print. Descriptive, not exact; keep tol small.
    """
    if trades.empty or other.empty:
        return trades
    delta = pd.Timedelta(tol).to_timedelta64()
    trade_ts = trades.index.values
    other_ts = other.index.values  # ascending
    pos = np.searchsorted(other_ts, trade_ts)
    left = np.clip(pos - 1, 0, len(other_ts) - 1)
    right = np.clip(pos, 0, len(other_ts) - 1)
    nearest = np.minimum(
        np.abs(trade_ts - other_ts[left]), np.abs(trade_ts - other_ts[right])
    )
    return trades[nearest > delta]


def trade_flow(
    trades: pd.DataFrame, freq: str = "500ms", index: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Net signed volume per bin (buyer- minus seller-initiated); empty bins 0.
    Pass index to align onto a shared grid. Cross-correlated directly (flow is
    already a rate, no differencing)."""
    if trades.empty:
        base = pd.Series(dtype=float)
    else:
        base = _signed_size(trades).resample(freq).sum()
    if index is not None:
        base = base.reindex(index, fill_value=0.0)
    base.name = "flow"
    return base


def trade_lead_lag_table(
    data: CalendarSpreadData,
    freq: str = "500ms",
    max_lag: int = 20,
    filter_implied: bool = True,
    coincidence_tol: str = "2ms",
    pairs: tuple = DEFAULT_TRADE_PAIRS,
) -> pd.DataFrame:
    """Lead-lag on executed-trade flow -- the price-discovery test the mids
    can't give (they're mechanically tied at lag 0). Signed volume per bin,
    cross-correlated. filter_implied strips leg prints coincident with a spread
    print first, so implied fills don't reintroduce the lag-0 tie."""
    front_t, back_t, spread_t = (
        data.front_trades, data.back_trades, data.spread_trades,
    )
    if filter_implied:
        front_t = drop_implied_coincident(front_t, data.spread_trades, coincidence_tol)
        back_t = drop_implied_coincident(back_t, data.spread_trades, coincidence_tol)

    present = [t for t in (front_t, back_t, spread_t) if not t.empty]
    cols = ["a", "b", "peak_lag", "peak_corr", "n", "interpretation"]
    if not present:
        return pd.DataFrame(columns=cols)

    # Align on the resampled bins (shared freq grid), not a date_range off the
    # first trade's raw timestamp -- that misses every bin and zeros the panel.
    flows = pd.concat(
        {
            "front_flow": trade_flow(front_t, freq),
            "back_flow": trade_flow(back_t, freq),
            "spread_flow": trade_flow(spread_t, freq),
        },
        axis=1,
    ).fillna(0.0)

    rows = []
    for a_col, b_col in pairs:
        res = lead_lag(flows[a_col], flows[b_col], max_lag)
        rows.append({"a": a_col, "b": b_col, "peak_lag": res["peak_lag"],
                     "peak_corr": res["peak_corr"], "n": float(len(flows)),
                     "interpretation": _interpret(a_col, b_col, res)})
    return pd.DataFrame(rows, columns=cols)


def front_volatility(data: CalendarSpreadData, spec: CalendarSpreadContractSpec) -> dict:
    """Front micro-price range and realized vol for the window, in ticks. Lets
    the scout rank days by front volatility (where the lead-lag question has
    signal), not just spread activity."""
    tick = float(spec.outright_tick_size)
    mid = _dedup_last(micro_price(data.front)).dropna()
    if len(mid) < 2:
        return {"front_range_ticks": float("nan"), "front_rv_ticks": float("nan"),
                "front_updates": int(len(data.front))}
    return {
        "front_range_ticks": float((mid.max() - mid.min()) / tick),
        "front_rv_ticks": float(mid.diff().dropna().std() / tick),
        "front_updates": int(len(data.front)),
    }
