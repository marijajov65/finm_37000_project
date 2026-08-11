# Calendar Spread Market-Making: ES (E-mini S&P 500)

Quantifying what it actually costs a market maker to quote the ES calendar spread — and whether CME needs to pay them to keep doing it.

## Results

Below is a live CME order-book snapshot for ESM6 (front), ESU6 (back), and the ESM6–ESU6 calendar spread, pulled directly from Databento at 2026-06-10 16:00:00 UTC (asks above bids; reproduce with `uv run src/main.py`):

```
        ESM6 (front)            ESU6 (back)          ESM6-ESU6 spread
       Sz         Px           Sz         Px           Sz         Px
     ------  ---------       ------  ---------       ------  ---------
         30    7341.50           9    7402.25          28      60.95   asks
         23    7341.25           7    7402.00         160      60.90
         19    7341.00           9    7401.75          20      60.85
         17    7340.75           7    7401.50         182      60.80
         12    7340.50 <ask      1    7401.25 <ask     128      60.75 <ask
     ------  ---------       ------  ---------       ------  ---------
         14    7340.00 <bid      4    7400.75 <bid       7      60.70 <bid
         15    7339.75           7    7400.50          45      60.65
         17    7339.50          16    7400.25          59      60.60
         20    7339.25          13    7400.00          20      60.55
         18    7339.00          13    7399.75          23      60.50   bids
```

A trader buying the spread pays the spread ask, 60.75. A market maker taking the other side and flattening immediately by legging out — selling ESM6 at the bid (7340.00) and buying ESU6 at the ask (7401.25) — pays 61.25 points to unwind a position it was paid only 60.75 for: a loss of 0.50 points, or **$25 per contract** at ES's $50/point multiplier.

That $25 floor is what the rest of the project tests against real sessions instead of one snapshot. Running the P&L model on two live 5-minute ES sessions (2026-06-09/10, 16:00 UTC, top-of-book, a 50% chance of winning a passive fill, a 5-contract inventory cap before legging out) gives:

| Scenario | Passive fee/rebate | Expected net P&L |
|---|---|---|
| Status quo (current $1.30 fee) | $1.30 | -$12,054 |
| Full fee waiver | $0.00 | -$9,931 |
| Break-even | **-$17.62** | $0 |

A fee waiver alone is not enough — the strategy still loses money at zero fees. Breaking even needs CME to actually pay the market maker about **$17.62 per spread contract** filled, not merely stop charging its own $1.30. Sweeping fill probability (5%–100%) and inventory tolerance (1–50 contracts) across a 6×6 grid puts the required rebate anywhere from **$6.95 to $22.71 per contract**, and it falls sharply as the inventory cap widens — fewer forced legging-outs is a much bigger lever than winning more passive fills:

| p (fill chance) \ cap | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| 0.05 | 21.47 | 19.85 | 16.74 | 14.21 | 10.79 | **6.95** |
| 0.25 | 21.62 | 20.39 | 18.09 | 16.02 | 12.57 | 10.08 |
| 0.50 | 22.69 | 21.37 | 18.98 | 16.97 | 13.81 | 11.71 |
| 1.00 | **22.71** | 21.49 | 19.17 | 17.01 | 14.09 | 11.86 |

On the deferred-leg side, front and back move almost in lockstep (correlation ≈0.98, contemporaneous, all three sampled sessions), and `deferred ≈ front + spread` fits extremely well as a regression (R² = 0.998–1.000, front coefficient ≈1.00). But the literal quoted touch is a different story: the implied deferred book matches the real ESU6 book's exact bid/ask only **3.5%–35%** of the time across the same three sessions (2026-06-12/15/16) — every one comes back **WEAK COUPLING**. Going the other direction is more striking: implying the spread from front + back produces a synthetic book **10–14 ticks wide** against the real spread market's **~1-tick** width — a roughly ten-times-wider synthetic spread, independently reproducing (from three live sessions three weeks after the snapshot above) the same ~10x tightness advantage the worked example opened with.

**Conclusion.** The legging-cost intuition holds up from two independent angles — a single order-book snapshot and a from-scratch book reconstruction three weeks later both put the spread's advantage at roughly 10x. But that cost is a floor, not the outcome: the P&L model shows the tested configuration losing money regardless, and closing the gap needs a real rebate, not just a fee cut. On the quoting side, `front + spread` is a strong *statistical* estimator of the deferred leg's fair value but a poor literal quoting rule — excellent fit in a regression sense, weak tick-for-tick — so a real quoting policy would need to smooth or lag the implied price rather than post it directly. These are five-minute windows on three or four sampled days, not a full backtest — directionally informative, not a final number.

## What's built

The project targets **ES only** (ESM6–ESU6); other CME products were dropped from scope to focus on one validated pipeline — see `DEVELOPER_NOTES.md` for the original plan. `legging_cost_calculator.py` and `pnl_calculator.py` replay real order-book and trade data through a fill model (passive-fill probability, inventory cap, CME's fee schedule) into the P&L above; `rebate_pivotality_analyzer.py` and `run_pivotality_inputs.py` solve it for the break-even rebate across a grid. `relationship_analysis.py` fits `deferred ≈ front + spread`; `implied_outright.py`/`implied_spread.py` build the synthetic books and score them against the real ones (`run_deferred_validator.py`, `run_strategy_analysis.py`). Both pipelines run end-to-end against live Databento data and are reproducible from the CLI below — this repository doesn't commit `data_cache/` or `results/`, so re-running refreshes the numbers above against whatever window you point them at.

## What's not there yet

No single interactive application or PDF report generator — each analysis is its own CLI script. Test coverage is solid for the market-data layer and the relationship module but thin on the P&L, rebate, and implied-book modules.

## Future work

- Investigate the touch-match / regression-fit gap on the deferred leg directly — why `deferred ≈ front + spread` fits so well in levels (R² ≈ 1.0) but rarely matches the real book tick-for-tick, and whether a smoothed or lagged implied quote closes it.
- Backtest the deferred-leg/spread quoting policy itself — not just validate the joint-book assumption — and report its P&L the way the bottom-up model above does for the spread.
- Widen the sample past a handful of 5-minute windows before treating any of the figures above as more than directional.
- Extend beyond ES once the pipeline has more mileage on it; `product_specs.json` already carries structural specs for ZN, CL, and 6E.
- Add unit tests for `pnl_calculator.py`, `rebate_pivotality_analyzer.py`, and the implied-book modules.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.12 (installed automatically by `uv` from `.python-version`)
- A [Databento](https://databento.com) account with access to the `GLBX.MDP3` (CME Globex) dataset, and an API key

## Setup

```bash
git clone <your-fork-url>
cd finm_37000_project
uv sync
```

Save your Databento API key as a single line in `~/.databento_api_key` (your home directory), with no quotes and no `KEY=` prefix — `src/util.py` reads it from that file, not from `.env`.

## Running the analysis

```bash
# Reproduce the worked-example order-book snapshot above
uv run src/main.py

# Goal 1: bottom-up P&L and rebate-pivotality pipeline (reproduces the Results table above)
uv run src/run_pnl_pipeline.py
uv run src/run_pivotality_inputs.py          # writes results/pivotality/, the grid table above
uv run src/run_rebate_analysis.py

# Goal 2: front/back/spread relationship and deferred-book validation
uv run src/run_relationship_analysis.py --full   # fetches and caches the analysis window
uv run src/run_deferred_validator.py             # reads the cache the previous command warmed
uv run src/run_strategy_analysis.py

# Tests
uv run pytest
```

First runs against a given day/window fetch from Databento and cache the result in `data_cache/` (git-ignored, since historical market data is metered); re-running the same window afterward is offline and free. Pass `--offline` to any script that supports it to error on a cache miss instead of re-fetching.

## Contributing

1. Create a branch on your fork for your changes.
2. Commit and push to your fork.
3. Open a pull request from your fork into `marijajov65/finm_37000_project` (`main` branch).
