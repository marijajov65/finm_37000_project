"""Run the lead-lag & co-movement analysis over the configured days (issue #11).

Prints only. Runs offline from the cached ES windows; an unseeded cache needs a
Databento key. Inputs come from analysis_config.py. Set AUTO_PICK_TOP > 0 to
analyze the busiest days from the scout instead of DAYS.

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
    resample_mids,
    spread_activity,
    spread_basis_sweep,
    spread_mid_diagnostic,
)
from analysis_config import (
    BOOK_LEVELS,
    DAYS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)

FREQ_SWEEP = ("100ms", "250ms", "500ms", "1s")
PRIMARY_FREQ = "500ms"
MAX_LAG = 40  # in grid steps; 40 * 500ms = 20s

# If > 0, re-scan the scout's candidate days and analyze the N busiest instead
# of DAYS. Costs a full re-scan, so off by default.
AUTO_PICK_TOP = 0


def _freq_sensitivity(data, pair=("front_mid", "back_mid")) -> pd.DataFrame:
    """front->back peak lag/corr at each grid in FREQ_SWEEP (the Epps check)."""
    rows = []
    for freq in FREQ_SWEEP:
        panel = resample_mids(data, freq=freq)
        row = lead_lag_table(panel, max_lag=MAX_LAG, pairs=(pair,)).iloc[0]
        rows.append({"freq": freq, "grid_points": len(panel),
                     "peak_lag_steps": row["peak_lag"], "peak_corr": round(float(row["peak_corr"]), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    print(f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})\n")

    if AUTO_PICK_TOP:
        from scout_spread_activity import top_active_days
        days = top_active_days(AUTO_PICK_TOP)
        print(f"auto-picked top {AUTO_PICK_TOP} by spread activity: {days}\n")
    else:
        days = DAYS

    for day in days:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        print(f"=== {day} {SESSION_START_UTC}..{t1:%H:%M:%S} UTC ===", flush=True)

        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)
        print(f"  events: front={len(data.front)} back={len(data.back)} spread={len(data.spread)}")

        panel = resample_mids(data, freq=PRIMARY_FREQ)
        implied = deferred_vs_implied(panel)
        act = spread_activity(data)
        mid = spread_mid_diagnostic(data, spec)

        print("\n  Liquidity:")
        print(liquidity_metrics(data, spec).round(3).to_string())
        print(f"\n  Change correlations ({PRIMARY_FREQ}):")
        print(change_corr(panel).round(3).to_string())
        print("\n  Lead-lag:")
        print(lead_lag_table(panel, max_lag=MAX_LAG).to_string(index=False))
        print("\n  Deferred ~ front + spread:")
        print(f"    R^2={implied['r_squared']:.4f}  beta_front={implied['beta_front']:.4f}"
              f"  beta_spread={implied['beta_spread']:.4f}  resid_std={implied['resid_std']:.4f}")
        print(f"\n  Spread activity: updates={act['spread_updates']} moves={act['spread_mid_moves']}"
              f" trades={act['spread_trades']} volume={act['spread_volume']:.0f}")
        print(f"  Spread mid: {mid['n_distinct_mids']} distinct, range {mid['mid_range_ticks']:.1f} ticks,"
              f" {mid['n_mid_moves']} moves (mean {mid['mean_move_ticks']:.2f})")
        print("\n  Spread vs basis by interval (peak_lag_s < 0 = basis leads):")
        print(spread_basis_sweep(data).to_string(index=False))
        print("\n  Front->back sensitivity:")
        print(_freq_sensitivity(data).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
