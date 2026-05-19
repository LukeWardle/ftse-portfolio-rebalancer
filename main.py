"""
main.py - FTSE 100 portfolio rebalancing pipelin.

"""
import sys
import numpy as np
from src.data import load_ftse_stocks, generate_correlated_returns
from src.ridge import ridge_regression, cross_validate_lambda
from src.constraints import project_weights, check_fca_compliance
from src.rebalance import generate_trade_list, estimate_transaction_costs
from src.analysis import compare_portfolios

def main() -> int:
  print("=" * 60)
  print("FTSE 100 Portfolio Rebalancing")
  print("=" * 60)

  # Step 1 — Load
  stocks = load_ftse_stocks()
  returns = generate_correlated_returns(stocks)
  y = np.mean(returns, axis=1)
  print(f"Loaded {len(stocks)} stocks, {len(returns)} days")

  # Step 2 — Optimise
  best_lam, _ = cross_validate_lambda(returns, y, [0.1, 1.0, 10.0])
  w_ridge = ridge_regression(returns, y, best_lam)
  print(f"Ridge: optimal lambda={best_lam}")

  # Step 3 — Constrain
  w_final = project_weights(w_ridge, stocks)
  compliance = check_fca_compliance(w_final, stocks)
  if not compliance["compliant"]:
    print("FCA VIOLATIONS:", compliance["violations"])
    return 1
  print("FCA compliance: PASS")

  # Step 4 — Rebalance
  current = np.ones(len(stocks)) / len(stocks)
  trades = generate_trade_list(current, w_final, stocks)
  costs = estimate_transaction_costs(trades)

  # Step 5 — Analyse and print
  comparison = compare_portfolios(current, w_final, returns, stocks)
  print(f"Trades: {costs['n_trades']} | Cost:   GBP{costs['total_cost']:,.0f}")
  print(f"Vol: {comparison['current_vol']:.1%} -> {comparison['optimised_vol']:.1%}")

  return 0

if __name__ == "__main__":
    sys.exit(main())