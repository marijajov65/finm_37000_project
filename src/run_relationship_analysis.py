"""Run the lead-lag & co-movement analysis end-to-end (Issue #11).

Wires the real cached fetcher to ``relationship_analysis``:

    CachedMarketDataFetcher -> CalendarSpreadData -> resample_mids ->
        liquidity_metrics / change_corr / lead_lag_table / deferred_vs_implied

Prints results only (no files). Runs offline against the cached ES windows;
only an unseeded cache needs a Databento key. Inputs (product, window, DAYS)
come from analysis_config.py; set AUTO_PICK_TOP to re-scan and analyze the
busiest days instead.

Run:
    uv run src/run_relationship_analysis.py
"""

from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from relationship_analysis import (
    change_corr,
    deferred_vs_implied,
    lead_lag_table,
    liquidity_metrics,
    liveliest_subwindow,
    resample_mids,
    spread_activity,
    spread_basis_sweep,
    spread_mid_diagnostic,
    spread_vs_basis,
)

# Shared fetch inputs (product, month, depth, window, days) live in one place so
# this script, the scout, and the tick probe stay cache-aligned. See
# analysis_config.py for the rationale behind each value.
from analysis_config import (  # noqa: E402
    BOOK_LEVELS,
    DAYS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

#: Sampling grids to sweep. The Epps effect attenuates cross-correlations at
#: very fine grids and the lead-lag smears out at coarse ones, so we report the
#: front->back peak lag/corr across a few and let the reader see the trade-off.
FREQ_SWEEP = ("100ms", "250ms", "500ms", "1s")
PRIMARY_FREQ = "500ms"

#: Max lag (in grid steps) to scan for lead-lag. At 500ms, 40 steps = 20s.
MAX_LAG = 40

#: Opt-in auto-selection. If > 0, re-scan the scout's CANDIDATE_DAYS and analyze
#: the N busiest-by-spread-activity days instead of the explicit DAYS from config.
#: Costs a full re-scan (metered on first pull) and makes the analyzed days
#: non-deterministic, so the default is 0 (off — use the explicit DAYS).
AUTO_PICK_TOP = 0


def _freq_sensitivity(data, pair=("front_mid", "back_mid")) -> pd.DataFrame:
    """Front->back peak lag/corr at each grid in FREQ_SWEEP (the Epps check)."""
    rows = []
    for freq in FREQ_SWEEP:
        panel = resample_mids(data, freq=freq)
        row = lead_lag_table(panel, max_lag=MAX_LAG, pairs=(pair,)).iloc[0]
        rows.append(
            {
                "freq": freq,
                "grid_points": len(panel),
                "peak_lag_steps": row["peak_lag"],
                "peak_corr": round(float(row["peak_corr"]), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    print(f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})\n")

    if AUTO_PICK_TOP:
        from scout_spread_activity import top_active_days
        days = top_active_days(AUTO_PICK_TOP)
        print(f"auto-picked top {AUTO_PICK_TOP} days by spread activity: {days}\n")
    else:
        days = DAYS

    for day in days:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        print(f"=== {day} {SESSION_START_UTC}..{t1:%H:%M:%S} UTC ===", flush=True)

        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)
        print(
            f"  book events: front={len(data.front)} back={len(data.back)} "
            f"spread={len(data.spread)}"
        )

        liq = liquidity_metrics(data, spec)
        panel = resample_mids(data, freq=PRIMARY_FREQ)
        corr = change_corr(panel)
        leadlag = lead_lag_table(panel, max_lag=MAX_LAG)
        implied = deferred_vs_implied(panel)
        sv_basis = spread_vs_basis(panel)
        activity = spread_activity(data)
        live = liveliest_subwindow(data, n_bins=6, metric="spread_trades")
        mid_diag = spread_mid_diagnostic(data, spec)
        sb_sweep = spread_basis_sweep(data)
        sens = _freq_sensitivity(data)

        print("\n  Liquidity:")
        print(liq.round(3).to_string())
        print(f"\n  Change correlations (grid={PRIMARY_FREQ}):")
        print(corr.round(3).to_string())
        print("\n  Lead-lag:")
        print(leadlag.to_string(index=False))
        print("\n  Deferred ~ front + spread:")
        print(
            f"    R^2={implied['r_squared']:.4f}  beta_front={implied['beta_front']:.4f}"
            f"  beta_spread={implied['beta_spread']:.4f}"
            f"  intercept={implied['intercept']:.4f}  resid_std={implied['resid_std']:.4f}"
        )
        print("\n  Spread vs basis (isolates the 'spread == basis' claim):")
        print(
            f"    R^2={sv_basis['r_squared']:.4f}  beta_spread={sv_basis['beta_spread']:.4f}"
            f"  change_corr={sv_basis['change_corr']:.4f}"
            f"  spread_moves={int(sv_basis['spread_n_changes'])}"
        )
        print(
            "\n  Spread activity (is this window worth trusting?): "
            f"updates={activity['spread_updates']} mid_moves={activity['spread_mid_moves']} "
            f"trades={activity['spread_trades']} volume={activity['spread_volume']:.0f}"
        )
        busy = pd.Timestamp(live["start"]).strftime("%H:%M:%S")
        print(
            f"    liveliest slice: bin {live['best_bin']}/{live['n_bins']} @ {busy} "
            f"({live['spread_trades']} spread trades, {live['spread_updates']} updates)"
        )
        print(
            "\n  Spread mid stickiness (why the spread tests stay weak): "
            f"{mid_diag['n_distinct_mids']} distinct mids, "
            f"range {mid_diag['mid_range_ticks']:.1f} ticks, "
            f"{mid_diag['n_mid_moves']} moves "
            f"(mean {mid_diag['move_size_ticks']['mean']:.2f} ticks)"
        )
        print(
            "\n  Spread vs basis by sampling interval "
            "(coarser cancels leg noise; peak_lag_s<0 => basis leads spread):"
        )
        print(sb_sweep.to_string(index=False))
        print("\n  Grid-frequency sensitivity (front->back):")
        print(sens.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
