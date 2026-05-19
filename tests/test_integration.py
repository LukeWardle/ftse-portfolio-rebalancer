"""
test_integration.py — Full pipeline integration test.

"""
import numpy as np
import pytest
from src.data import load_ftse_stocks, generate_correlated_returns
from src.ridge import ridge_regression, cross_validate_lambda
from src.constraints import project_weights, check_fca_compliance
from src.rebalance import generate_trade_list, estimate_transaction_costs
from src.analysis import compare_portfolios

def test_full_pipeline():
  """
  Full pipeline — load, optimise, constrain, rebalance, analyse.
  
  """
  # Load
  stocks = load_ftse_stocks()
  returns = generate_correlated_returns(stocks)
  y = np.mean(returns, axis=1)

  # Optimise
  best_lam, _ = cross_validate_lambda(returns, y, [0.1, 1.0, 10.0])
  w_ridge = ridge_regression(returns, y, best_lam)

  # Constrain
  w_final = project_weights(w_ridge, stocks)
  compliance = check_fca_compliance(w_final, stocks)
  assert compliance["compliant"], f"FCA violations: {compliance['violations']}"

  # Rebalance
  current = np.ones(len(stocks)) / len(stocks)
  trades = generate_trade_list(current, w_final, stocks)
  costs = estimate_transaction_costs(trades)

  # Analyse
  comparison = compare_portfolios(current, w_final, returns, stocks)
  assert comparison["current_vol"] > 0
  assert comparison["optimised_vol"] > 0
  assert abs(sum(comparison["current_sector"].values()) - 1.0) < 1e-6
  assert abs(sum(comparison["optimised_sector"].values()) - 1.0) < 1e-6