"""
rebalance.py - Trade list generation and UK transaction cost estimation

Generate executable trade instructions from weight deltas and estimate UK transaction costs including SORT stamp duty.

"""
import numpy as np
from typing import List, Dict

def generate_trade_list(current_weights: np.ndarray,
                        target_weights: np.ndarray,
                        stocks: List[Dict],
                        portfolio_value: float = 50_000_000) -> List[Dict]:
  """
  Generate executable trade list from weight deltas.
  Only generates trades where |delta| > 1e-4 (10 bps threshold).

  """
  trades = []
  for i, stock in enumerate(stocks):
    delta = target_weights[i] - current_weights[i]
    if abs(delta) > 1e-4:
      trades.append({
        "ticker": stock["ticker"],
        "action": "BUY" if delta > 0 else "SELL",
        "amount_gbp": abs(delta) * portfolio_value,
        "weight_change": delta,
      })
  trades.sort(key=lambda t: t["amount_gbp"], reverse=True)
  return trades

def estimate_transaction_costs(trades: List[Dict],
                               fixed_cost: float = 25.0,
                               commission_rate: float = 0.001) -> Dict:
  """
  Estimate UK transaction costs.
  Fixed: £25 per trade
  Commission: 0.1% of trade value
  Stamp duty: 0.5% on BUY trades only (SDRT for UK equities)

  """
  total_fixed = len(trades) * fixed_cost
  commission = sum(t["amount_gbp"] * commission_rate for t in trades)
  stamp_duty = sum(t["amount_gbp"] * 0.005 for t in trades if t["action"] == "BUY")
  total = total_fixed + commission + stamp_duty
  traded = sum(t["amount_gbp"] for t in trades) or 1
  return {
    "n_trades": len(trades),
    "fixed_costs": total_fixed,
    "commission": commission,
    "stamp_duty": stamp_duty,
    "total_cost": total,
    "cost_pct": total / traded,
  }