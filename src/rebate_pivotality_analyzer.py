"""RebatePivotalityAnalyzer: exchange incentive optimization framework (Issue #10).

Ingests standardized P&L metrics from Issue #9 and computes the exact exchange 
incentives required to maintain a break-even P&L. Evaluates a fee waiver and the
exact break-even rebate, and cross-references against real-world CME programs.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

class RebatePivotalityAnalyzer:
    """Executes the exchange rebate pivotality analysis and feasibility check."""

    def __init__(self, current_passive_fee: float, current_aggressive_fee: float, daily_fixed_costs: float = 0.0):
        """
        Initializes the analyzer with baseline fees and infrastructure costs.
        (Negative passive_fee denotes a per-contract rebate).
        """
        self.current_passive_fee = current_passive_fee
        self.current_aggressive_fee = current_aggressive_fee
        self.daily_fixed_costs = daily_fixed_costs

    def _get_average_metrics(self, pnl_samples: List[pd.Series]) -> pd.Series:
        """Helper to average the Bernoulli-driven simulation seeds from Issue #9."""
        return pd.DataFrame(pnl_samples).mean()

    def generate_pivotality_report(
        self, 
        standard_pnl_samples: List[pd.Series], 
        dmm_pnl_samples: Optional[List[pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Generates the comparative pivotality report detailing required rebate levels.
        Optionally ingests DMM priority-matching samples (higher queue-head probability) 
        to evaluate allocation benefits side-by-side.
        """
        results: List[Dict[str, Any]] = []
        
        # 1. Process Standard Market Maker Flow
        std_avg = self._get_average_metrics(standard_pnl_samples)
        self._append_scenarios(results, std_avg, flow_type="Standard Flow")

        # 2. Process DMM Priority Matching Flow (if provided)
        if dmm_pnl_samples is not None:
            dmm_avg = self._get_average_metrics(dmm_pnl_samples)
            self._append_scenarios(results, dmm_avg, flow_type="DMM Priority Matching")

        report_df = pd.DataFrame(results)
        for col in ["Passive Fee/Rebate", "Expected Net P&L", "Gross Trading P&L"]:
            report_df[col] = report_df[col].round(4)
            
        return report_df

    def _append_scenarios(self, results: List[Dict[str, Any]], avg_metrics: pd.Series, flow_type: str):
        """Computes all support mechanisms and solves for break-even targets."""
        filled = avg_metrics["filled_contracts"]
        hedged = avg_metrics["hedged_contracts"]
        
        # Isolate Gross Trading PnL (pre-exchange fees and pre-fixed costs)
        pure_legging_costs = avg_metrics["hedge_costs"] - (self.current_aggressive_fee * hedged)
        gross_pnl = avg_metrics["cash"] + avg_metrics["position_mark"] - pure_legging_costs

        # --- Scenario A: Status Quo ---
        sq_net = gross_pnl - (self.current_passive_fee * filled) - (self.current_aggressive_fee * hedged) - self.daily_fixed_costs
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Status Quo",
            "Support Mechanism": "None",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": self.current_passive_fee,
            "Expected Net P&L": sq_net
        })

        # --- Scenario B: Total Fee Waiver ---
        waiver_net = gross_pnl - self.daily_fixed_costs
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Total Fee Waiver",
            "Support Mechanism": "Zero Trans. Fees",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": 0.0,
            "Expected Net P&L": waiver_net
        })

        # --- Scenario C: Exact Break-Even Solver ---
        be_fee = (gross_pnl - (self.current_aggressive_fee * hedged) - self.daily_fixed_costs) / filled if filled > 0 else 0.0
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Target Break-Even",
            "Support Mechanism": "Rebate Only",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": be_fee,
            "Expected Net P&L": 0.0
        })