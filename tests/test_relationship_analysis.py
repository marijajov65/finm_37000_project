"""Tests for relationship_analysis (Issue #11).

All synthetic, numpy + pandas only. The lead-lag tests plant a KNOWN lag
(build the back book as a delayed copy of the front) and assert the analysis
recovers it, and the co-movement test builds an exact deferred = front + spread
identity and asserts the regression finds R^2 = 1 with unit slopes.
"""

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from market_data_fetcher import (  # noqa: E402
    BOOK_INDEX_NAME,
    CalendarSpreadContractSpec,
    CalendarSpreadData,
)
import relationship_analysis as ra  # noqa: E402


# --------------------------------------------------------------------------- #
# Builders (mbp-1 shape: a single top-of-book level)
# --------------------------------------------------------------------------- #
def _grid(n, freq="500ms", start="2026-06-09T16:00:00Z"):
    idx = pd.date_range(start=start, periods=n, freq=freq)
    idx.name = BOOK_INDEX_NAME
    return idx


def _book(mids, idx, half_spread=0.125, bid_sz=10.0, ask_sz=10.0):
    """One-level book with the given mids and a symmetric quoted spread."""
    mids = np.asarray(mids, dtype=float)
    return pd.DataFrame(
        {
            "bid_px_00": mids - half_spread,
            "ask_px_00": mids + half_spread,
            "bid_sz_00": bid_sz,
            "ask_sz_00": ask_sz,
        },
        index=idx,
    )


def _trades(idx, prices, sizes=None, sides=None):
    n = len(prices)
    return pd.DataFrame(
        {
            "price": prices,
            "size": sizes if sizes is not None else [1.0] * n,
            "side": sides if sides is not None else ["B"] * n,
        },
        index=idx,
    )


def _spec():
    return CalendarSpreadContractSpec(
        product_code="ES",
        front_symbol="ESM6",
        back_symbol="ESU6",
        spread_symbol="ESM6-ESU6",
        outright_tick_size=Decimal("0.25"),
        spread_tick_size=Decimal("0.05"),
        contract_multiplier=Decimal("50"),
    )


def _data(front_book, back_book, spread_book, empty_trades_idx=None):
    ti = empty_trades_idx if empty_trades_idx is not None else _grid(1)
    empty = _trades(ti[:0], [], [], [])
    return CalendarSpreadData(
        front_symbol="ESM6",
        back_symbol="ESU6",
        spread_symbol="ESM6-ESU6",
        front=front_book,
        back=back_book,
        spread=spread_book,
        front_trades=empty,
        back_trades=empty,
        spread_trades=empty,
    )


# --------------------------------------------------------------------------- #
# top_mid / resample_mids
# --------------------------------------------------------------------------- #
def test_top_mid():
    idx = _grid(3)
    book = _book([100.0, 101.0, 102.0], idx, half_spread=0.25)
    mid = ra.top_mid(book)
    assert list(mid) == [100.0, 101.0, 102.0]


def test_resample_mids_aligns_and_derives():
    idx = _grid(5)
    front = _book([7340.0, 7340.5, 7341.0, 7341.0, 7341.5], idx)
    back = _book([7401.0, 7401.5, 7402.0, 7402.0, 7402.5], idx)
    spread = _book([60.9, 60.9, 61.0, 61.0, 61.0], idx)
    panel = ra.resample_mids(_data(front, back, spread), freq="500ms")

    assert list(panel.columns) == [
        "front_mid", "back_mid", "spread_mid", "basis", "implied_deferred",
    ]
    # basis = back - front, implied_deferred = front + spread
    np.testing.assert_allclose(panel["basis"], panel["back_mid"] - panel["front_mid"])
    np.testing.assert_allclose(
        panel["implied_deferred"], panel["front_mid"] + panel["spread_mid"]
    )


def test_resample_mids_ffills_across_unaligned_clocks():
    # front updates every 500ms; back only twice. Back must carry forward.
    fidx = _grid(4, freq="500ms")
    bidx = pd.DatetimeIndex(
        pd.to_datetime(
            ["2026-06-09T16:00:00.000Z", "2026-06-09T16:00:01.500Z"], format="ISO8601"
        ),
        name=BOOK_INDEX_NAME,
    )
    front = _book([100.0, 101.0, 102.0, 103.0], fidx)
    back = _book([200.0, 205.0], bidx)
    spread = _book([1.0, 1.0, 1.0, 1.0], fidx)
    panel = ra.resample_mids(_data(front, back, spread), freq="500ms")
    # back holds 200 until its second event at t=1.5s, then 205
    assert panel["back_mid"].iloc[0] == 200.0
    assert panel["back_mid"].iloc[-1] == 205.0
    assert not panel["back_mid"].isna().any()


def test_resample_mids_rejects_empty_book():
    idx = _grid(3)
    good = _book([100.0, 101.0, 102.0], idx)
    empty = good.iloc[:0]
    with pytest.raises(ValueError):
        ra.resample_mids(_data(good, empty, good))


# --------------------------------------------------------------------------- #
# Lead-lag: recover a planted lag
# --------------------------------------------------------------------------- #
def test_cross_correlation_recovers_planted_lag():
    rng = np.random.default_rng(0)
    n, k = 400, 3
    steps = rng.normal(0, 1, n)
    front_mid = 7000 + np.cumsum(steps)
    # back is the front delayed by k steps: back[t] = front[t-k]
    back_mid = np.empty(n)
    back_mid[:k] = front_mid[0]
    back_mid[k:] = front_mid[:-k]

    idx = _grid(n)
    front = _book(front_mid, idx)
    back = _book(back_mid, idx)
    spread = _book(np.full(n, 60.0), idx)
    panel = ra.resample_mids(_data(front, back, spread), freq="500ms")

    df = panel[["front_mid", "back_mid"]].diff().dropna()
    ccf = ra.cross_correlation(df["front_mid"], df["back_mid"], max_lag=10)
    assert int(ccf.abs().idxmax()) == k  # peaks at +k => front leads

    res = ra.lead_lag(df["front_mid"], df["back_mid"], max_lag=10)
    assert res["peak_lag"] == k
    assert res["leads"] == "a"
    assert res["peak_corr"] > 0.99


def test_lead_lag_table_reports_front_leading():
    rng = np.random.default_rng(1)
    n, k = 300, 2
    steps = rng.normal(0, 1, n)
    front_mid = 7000 + np.cumsum(steps)
    back_mid = np.concatenate([[front_mid[0]] * k, front_mid[:-k]])
    spread_mid = 60 + np.cumsum(rng.normal(0, 0.02, n))  # non-degenerate
    idx = _grid(n)
    panel = ra.resample_mids(
        _data(_book(front_mid, idx), _book(back_mid, idx), _book(spread_mid, idx)),
        freq="500ms",
    )
    table = ra.lead_lag_table(panel, max_lag=8)
    fb = table[(table["a"] == "front_mid") & (table["b"] == "back_mid")].iloc[0]
    assert fb["peak_lag"] == k
    assert "front_mid leads back_mid" in fb["interpretation"]


def test_cross_correlation_contemporaneous_for_identical_series():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(0, 1, 200))
    ccf = ra.cross_correlation(x, x.copy(), max_lag=5)
    assert int(ccf.idxmax()) == 0
    assert ccf.loc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Co-movement: exact identity
# --------------------------------------------------------------------------- #
def test_deferred_vs_implied_exact_identity():
    rng = np.random.default_rng(3)
    n = 250
    front_mid = 7000 + np.cumsum(rng.normal(0, 1, n))
    spread_mid = 60 + np.cumsum(rng.normal(0, 0.05, n))
    back_mid = front_mid + spread_mid  # deferred == front + spread, exactly

    idx = _grid(n)
    panel = ra.resample_mids(
        _data(_book(front_mid, idx), _book(back_mid, idx), _book(spread_mid, idx)),
        freq="500ms",
    )
    fit = ra.deferred_vs_implied(panel)
    assert fit["r_squared"] > 0.9999
    assert fit["beta_front"] == pytest.approx(1.0, abs=1e-6)
    assert fit["beta_spread"] == pytest.approx(1.0, abs=1e-6)
    assert fit["intercept"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Liquidity
# --------------------------------------------------------------------------- #
def test_liquidity_metrics_shape_and_values():
    idx = _grid(4)
    front = _book([7340.0] * 4, idx, half_spread=0.125, bid_sz=100.0, ask_sz=100.0)
    back = _book([7401.0] * 4, idx, half_spread=0.125, bid_sz=80.0, ask_sz=80.0)
    spread = _book([60.9] * 4, idx, half_spread=0.025, bid_sz=300.0, ask_sz=300.0)
    tidx = _grid(2)
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=front, back=back, spread=spread,
        front_trades=_trades(tidx, [7340.0, 7340.0], [5.0, 5.0]),
        back_trades=_trades(tidx[:0], [], []),
        spread_trades=_trades(tidx, [60.9, 60.9, 60.9][:2], [10.0, 10.0]),
    )
    liq = ra.liquidity_metrics(data, _spec())

    assert list(liq.index) == ["front", "back", "spread"]
    # front quoted spread = 0.25 pts = 1 outright tick = $12.50
    assert liq.loc["front", "avg_spread_pts"] == pytest.approx(0.25)
    assert liq.loc["front", "avg_spread_ticks"] == pytest.approx(1.0)
    assert liq.loc["front", "avg_spread_usd"] == pytest.approx(12.5)
    # spread quoted spread = 0.05 pts = 1 spread tick = $2.50
    assert liq.loc["spread", "avg_spread_ticks"] == pytest.approx(1.0)
    assert liq.loc["spread", "avg_spread_usd"] == pytest.approx(2.5)
    # depth + trade counts
    assert liq.loc["spread", "avg_top_depth"] == pytest.approx(300.0)
    assert liq.loc["front", "n_trades"] == 2.0
    assert liq.loc["back", "n_trades"] == 0.0


# --------------------------------------------------------------------------- #
# spread_vs_basis
# --------------------------------------------------------------------------- #
def test_spread_vs_basis_recovers_identity_when_spread_moves():
    # basis == spread exactly, and the spread genuinely varies -> clean fit.
    rng = np.random.default_rng(10)
    n = 300
    front_mid = 7000 + np.cumsum(rng.normal(0, 1, n))
    spread_mid = 60 + np.cumsum(rng.normal(0, 0.1, n))  # moves meaningfully
    back_mid = front_mid + spread_mid                   # basis == spread
    idx = _grid(n)
    panel = ra.resample_mids(
        _data(_book(front_mid, idx), _book(back_mid, idx), _book(spread_mid, idx)),
        freq="500ms",
    )
    res = ra.spread_vs_basis(panel)
    assert res["beta_spread"] == pytest.approx(1.0, abs=1e-6)
    assert res["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert res["r_squared"] > 0.9999
    assert res["change_corr"] == pytest.approx(1.0, abs=1e-6)
    assert res["spread_n_changes"] > 0


def test_spread_vs_basis_flags_a_static_spread():
    # A near-constant spread: the flag should show it barely moved, even if
    # the level fit still reports something.
    rng = np.random.default_rng(11)
    n = 200
    front_mid = 7000 + np.cumsum(rng.normal(0, 1, n))
    spread_mid = np.full(n, 60.0)          # never moves
    back_mid = front_mid + spread_mid
    idx = _grid(n)
    panel = ra.resample_mids(
        _data(_book(front_mid, idx), _book(back_mid, idx), _book(spread_mid, idx)),
        freq="500ms",
    )
    res = ra.spread_vs_basis(panel)
    assert res["spread_n_changes"] == 0     # the caveat flag fires


# --------------------------------------------------------------------------- #
# spread_activity / liveliest_subwindow
# --------------------------------------------------------------------------- #
def test_spread_activity_counts_moves_and_trades():
    idx = _grid(5)
    spread = _book([60.0, 60.0, 60.5, 60.5, 61.0], idx)  # 2 distinct moves
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=_book([7340.0] * 5, idx), back=_book([7401.0] * 5, idx), spread=spread,
        front_trades=_trades(idx[:0], [], []),
        back_trades=_trades(idx[:0], [], []),
        spread_trades=_trades(_grid(3), [60.0, 60.5, 61.0], [10.0, 20.0, 30.0]),
    )
    act = ra.spread_activity(data)
    assert act["spread_updates"] == 5
    assert act["spread_mid_moves"] == 2
    assert act["spread_trades"] == 3
    assert act["spread_volume"] == 60.0


def test_liveliest_subwindow_finds_the_busy_slice():
    # 60 timestamps; put all the spread trades in the last third.
    idx = _grid(60, freq="1s")
    spread = _book(60.0 + np.zeros(60), idx)
    trade_idx = idx[40:50]  # trades concentrated late in the window
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=_book([7340.0] * 60, idx), back=_book([7401.0] * 60, idx), spread=spread,
        front_trades=_trades(idx[:0], [], []),
        back_trades=_trades(idx[:0], [], []),
        spread_trades=_trades(trade_idx, [60.0] * 10, [1.0] * 10),
    )
    res = ra.liveliest_subwindow(data, n_bins=6, metric="spread_trades")
    # bins are ~10s each; trades land in bin index 4 (seconds 40-49)
    assert res["best_bin"] == 4
    assert res["spread_trades"] == 10
    assert len(res["bins"]) == 6


# --------------------------------------------------------------------------- #
# spread_mid_diagnostic
# --------------------------------------------------------------------------- #
def test_spread_mid_diagnostic_reports_stickiness():
    # A spread that trades a lot but takes only 3 distinct mids over the window.
    idx = _grid(10)
    spread_mids = [60.90, 60.90, 60.90, 60.95, 60.95, 60.90, 60.90, 60.95, 61.00, 61.00]
    spread = _book(spread_mids, idx, half_spread=0.025)
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=_book([7340.0] * 10, idx), back=_book([7401.0] * 10, idx), spread=spread,
        front_trades=_trades(idx[:0], [], []),
        back_trades=_trades(idx[:0], [], []),
        spread_trades=_trades(idx, [60.9] * 10, [50.0] * 10),  # lots of volume...
    )
    diag = ra.spread_mid_diagnostic(data, _spec())
    assert diag["n_distinct_mids"] == 3               # ...but only 3 distinct mids
    assert diag["distinct_mids"] == [60.90, 60.95, 61.00]
    assert diag["mid_range_ticks"] == pytest.approx(2.0)   # (61.00-60.90)/0.05
    assert diag["n_mid_moves"] == 4
    assert diag["move_size_ticks"]["mean"] == pytest.approx(1.0)  # each move = 1 tick


# --------------------------------------------------------------------------- #
# zero-variance guard (no divide warnings) + spread_basis_sweep
# --------------------------------------------------------------------------- #
def test_change_corr_constant_column_is_nan_without_warning():
    # A constant spread column must yield NaN correlations, not a RuntimeWarning.
    idx = _grid(50)
    front = _book(7000 + np.arange(50) * 0.1, idx)
    back = _book(7060 + np.arange(50) * 0.1, idx)
    spread = _book([60.0] * 50, idx)  # never moves -> zero variance
    panel = ra.resample_mids(_data(front, back, spread), freq="500ms")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # any divide warning -> test fails
        corr = ra.change_corr(panel)
    assert np.isnan(corr.loc["spread_mid", "front_mid"])   # NaN, cleanly


def test_spread_basis_sweep_recovers_a_planted_lead_at_coarse_grid():
    # Build basis that LEADS the spread by a known amount, then confirm the
    # sweep reports a negative peak_lag (basis leads) with real correlation
    # at a coarse grid.
    rng = np.random.default_rng(20)
    n = 6000
    idx = _grid(n, freq="100ms")  # 600s of 100ms data
    front = 7000 + np.cumsum(rng.normal(0, 0.1, n))
    basis_true = 60 + np.cumsum(rng.normal(0, 0.05, n))
    back = front + basis_true                       # basis = back - front
    lead = 30                                       # spread lags basis by 30*100ms = 3s
    spread_mid = np.concatenate([[basis_true[0]] * lead, basis_true[:-lead]])
    panel_data = _data(_book(front, idx), _book(back, idx), _book(spread_mid, idx))

    sweep = ra.spread_basis_sweep(panel_data, freqs=("500ms", "5s"), max_lag_seconds=30)
    row5s = sweep[sweep["freq"] == "5s"].iloc[0]
    # basis leads spread -> negative lag; correlation should be clearly non-zero
    assert row5s["peak_lag_s"] < 0
    assert abs(row5s["peak_corr"]) > 0.3
    assert list(sweep["freq"]) == ["500ms", "5s"]
