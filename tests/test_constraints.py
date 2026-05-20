"""
test_constraints.py - Test suite for FCA constraint projection

"""
import numpy as np
import pytest
from src.constraints import project_weights, check_fca_compliance

def make_stocks(n=40):
  """
  Minimal stock list for testing.

  """
  sectors = ["A", "B", "C", "D"]
  return [{"sector": sectors[i % 4], "liquid": i < 5} for i in range(n)]

def test_weights_sum_to_one():
  """
  project_weights() must return weights summing to 1.0.
  
  """
  w_raw = np.random.randn(40)
  w = project_weights(w_raw, make_stocks())
  assert abs(w.sum() - 1.0) < 1e-4

def test_no_negative_weights():
  """
  project_weights() must remove all negative weights.
  
  """
  w_raw = np.random.randn(40)
  w = project_weights(w_raw, make_stocks())
  assert (w >= -1e-4).all()

def test_stock_limit_enforced():
  """
  project_weights() must cap every stock at 10%.
  
  """
  w_raw = np.random.randn(40)
  w = project_weights(w_raw, make_stocks())
  assert (w <= 0.10 + 1e-4).all(), f"Max stock = {w.max():.3f}"

def test_sector_limit_enforced():
  """
  project_weights() must cap every sector at 30%.
  
  """
  stocks = make_stocks(40)
  w_raw = np.random.randn(40)
  w = project_weights(w_raw, stocks)
  for sec in set(s["sector"] for s in stocks):
    idx = [i for i, s in enumerate(stocks) if s["sector"] == sec]
    assert w[idx].sum() <= 0.30 + 1e-4

def test_fca_compliance_checker_passes_valid():
  """
  check_fca_compliance() must return compliant on valid weights.
  
  """
  stocks = make_stocks(40)
  w = project_weights(np.random.randn(40), stocks)
  result = check_fca_compliance(w, stocks)
  assert result["compliant"], f"Violations: {result['violations']}"

def test_liquidity_minimum_enforced():
  """
  FCA: liquid stocks must receive at least 5% total weight after projection.
  
  """
  stocks = make_stocks(40)  # 5 liquid stocks (i < 5)

  # Force all weight onto illiquid stocks
  w_raw = np.zeros(40)
  w_raw[10:] = 1.0  # all illiquid
  w = project_weights(w_raw, stocks)
  liquid_idx = [i for i, s in enumerate(stocks) if s["liquid"]]
  assert w[liquid_idx].sum() >= 0.05 - 1e-4

  # Also verify stock cap not violated
  assert (w <= 0.10 + 1e-4).all()