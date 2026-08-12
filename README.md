# Calendar Spread Market-Making: ES (E-mini S&P 500)

Quantifying what it actually costs a market maker to quote the ES calendar spread — and whether CME needs to pay them to keep doing it.

## Results

CME order books for ESM6 (front), ESU6 (back) and the ESM6–ESU6 spread, from Databento at 2026-06-10 16:00 UTC. (asks above bids; reproduce with `uv run src/main.py`):

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

A trader buying the spread pays its offer, 60.75, leaving the market maker short. Flattening immediately — selling ESM6 at its bid (7340.00), buying ESU6 at its offer (7401.25) — costs 61.25 points to unwind a position it was paid 60.75 for: a loss of 0.50 points, or **$25 per contract** at ES's $50/point multiplier.

That $25 is the cost of hedging one fill instantly — a floor on what quoting costs, not a forecast of what a desk earns. Turning it into a P&L means replaying real sessions: two 5-minute ES windows (2026-06-09/10, 16:00 UTC), best bid and offer only, a 50% chance of winning any passive fill, a 5-contract inventory cap before legging out:

| Scenario | Passive fee/rebate | Expected net P&L |
|---|---|---|
| Status quo (current $1.30 fee) | $1.30 | -$12,054 |
| Full fee waiver | $0.00 | -$9,931 |
| Break-even | **-$17.62** | $0 |

The strategy still loses at zero fees, so waiving the fee does not close the gap. Break-even needs CME to pay the maker roughly **$17.62 per spread contract filled** — a genuine rebate, not a discount on the $1.30 it charges today.

Sweeping fill probability (5%–100%) and inventory tolerance (1–50 contracts) across a 6×6 grid puts the required rebate anywhere from **$6.95 to $22.71 per contract**, and it falls sharply as the inventory cap widens — fewer forced legging-outs is a much bigger lever than winning more passive fills:

| p (fill chance) \ cap | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| 0.05 | 21.47 | 19.85 | 16.74 | 14.21 | 10.79 | **6.95** |
| 0.25 | 21.62 | 20.39 | 18.09 | 16.02 | 12.57 | 10.08 |
| 0.50 | 22.69 | 21.37 | 18.98 | 16.97 | 13.81 | 11.71 |
| 1.00 | **22.71** | 21.49 | 19.17 | 17.01 | 14.09 | 11.86 |

On the deferred-leg side, front and back move almost in lockstep (correlation ≈0.98, contemporaneous, all three sampled sessions), and `deferred ≈ front + spread` fits extremely well as a regression (R² = 0.998–1.000, front coefficient ≈1.00). But the literal quoted touch is a different story: the implied deferred book matches the real ESU6 book's exact bid/ask only **3.5%–35%** of the time across the same three sessions (2026-06-12/15/16), failing the validator's 90% **WEAK COUPLING** threshold every session. The reason is structural: the modeled quote inherits the front leg's width plus the spread's, then rounds outward onto the coarser 0.25 grid — about a tick wider than the real book, too wide to sit on both sides at once.

Running the construction backwards is starker. A spread implied from its two legs is **10–14 spread ticks wide** against a real market quoting **~1 tick**. An implied spread is exactly as wide as its two legs combined, and ES legs trade in quarter-points while the spread trades in nickels. The spread's tightness cannot be rebuilt from its own legs — which rules out quoting it off them, and leaves any maker filled on the spread without a synthetic exit near its own quote.

**Conclusion.** One piece of arithmetic explains both halves of the project: a synthetic quote is as wide as everything it is built from. For the spread that is fatal — its own legs imply a market ten times too wide to compete, so the maker quoting the real thing pays that difference every time it unwinds. That legging cost, not the exchange fee, is what sinks the P&L: the tested configuration loses money even at zero fees, so closing the gap takes a genuine rebate rather than a discount. The sweep points the same way, barely moving with fill probability while falling by half or more as the inventory cap widens — forced unwinding, not lost queue position, is the cost. For the deferred leg the same arithmetic is merely inconvenient: front + spread lands about a tick wider than the book it competes with, enough to estimate fair value and not enough to post a price. All of this rests on five-minute windows across five days: directional, not a backtest.

## What's built

The project targets **ES only** (ESM6–ESU6); other CME products were dropped from scope to focus on one validated pipeline — see `DEVELOPER_NOTES.md` for the original plan. `legging_cost_calculator.py` and `pnl_calculator.py` replay real order-book and trade data through a fill model (passive-fill probability, inventory cap, CME's fee schedule) into the P&L above; `rebate_pivotality_analyzer.py` and `run_pivotality_inputs.py` solve it for the break-even rebate across a grid. `relationship_analysis.py` fits `deferred ≈ front + spread`; `implied_outright.py`/`implied_spread.py` build the synthetic books and score them against the real ones (`run_deferred_validator.py`, `run_strategy_analysis.py`). Both pipelines run end-to-end against live Databento data and are reproducible from the CLI below — this repository doesn't commit `data_cache/` or `results/`, so re-running refreshes the numbers above against whatever window you point them at.

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
