"""
test_rebalance.py - Test suite for trade list generation

"""
import numpy as np
import pytest
from src.rebalance import generate_trade_list, estimate_transaction_costs

def test_trades_generated_for_weight_changes():
  """
  generate_trade_list() must produce trades when weights differ.
  
  """
  stocks = [{"ticker": f"S{i}", "liquid": True} for i in range(5)]
  current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
  target = np.array([0.10, 0.30, 0.20, 0.15, 0.25])
  trades = generate_trade_list(current, target, stocks)
  assert len(trades) > 0
  assert all("ticker" in t for t in trades)

def test_trade_directions_correct():
    """
    generate_trade_list() must assign BUY/SELL correctly.
    
    """
    stocks = [{"ticker": "A", "liquid": True},
              {"ticker": "B", "liquid": True}]
    current = np.array([0.40, 0.60])
    target = np.array([0.60, 0.40])
    trades = generate_trade_list(current, target, stocks)
    trade_map = {t["ticker"]: t["action"] for t in trades}
    assert trade_map["A"] == "BUY"
    assert trade_map["B"] == "SELL"

def test_cost_estimate_reasonable():
  """
  estimate_transaction_costs() must return a positive cost breakdown.
  
  """
  stocks = [{"ticker": f"S{i}", "liquid": True} for i in range(5)]
  current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
  target = np.array([0.10, 0.30, 0.20, 0.15, 0.25])
  trades = generate_trade_list(current, target, stocks)
  costs = estimate_transaction_costs(trades)
  assert costs["n_trades"] > 0
  assert costs["total_cost"] > 0
  assert costs["stamp_duty"] >= 0
  assert costs["cost_pct"] < 0.10