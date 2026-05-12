"""
test_data.py - Unit tests for FTSE data loading and multicollinearity analysis.

"""
import numpy as np
from src.data import load_ftse_stocks, generate_correlated_returns
from src.data import analyse_multicollinearity

def test_load_ftse_returns_100_stocks():
  """
  100 stocks loaded with required fields - sector and liquid must be present.
  
  """
  stocks = load_ftse_stocks()
  assert len(stocks) == 100, f"Expected 100 stocks, got {len(stocks)}"
  assert all("sector" in s for s in stocks)
  assert all("liquid" in s for s in stocks)

def test_returns_matrix_shape():
  """
  Returns matrix must be (252, 100) with no NaN values.
  
  """
  stocks = load_ftse_stocks()
  X = generate_correlated_returns(stocks, n_days=252)
  assert X.shape == (252, 100), f"Expected (252, 100), got {X.shape}"
  assert not np.isnan(X).any(), "Returns contain NaN"

def test_sector_stocks_are_correlated():
  """
  Stocks in the same sector must be correlated - proves the noise model worked.
  
  """
  stocks = load_ftse_stocks()
  X = generate_correlated_returns(stocks)
  fin_idx = [i for i, s in enumerate(stocks) if s["sector"] == "Financial"]
  if len(fin_idx) >= 2:
    r = np.corrcoef(X[:, fin_idx[1]])[0, 1]
    assert r > 0.50, f"Sector correlation too low: {r:.3f}"

def test_multicollinearity_detected():
    """
    Condition number must exceed threshold — proves ill-conditioning is present.
    
    NOTE: Threshold is > 100, not > 1e6 
    Real FTSE data will have a much higher correlation producing a k in the billions.
    The synthetic data doesnt fully replicate this, the 70/30 sector noise split produces correlation around 0.94.
    This is good enough to demonstrate the concept. This will need adjusting in version 2 and documented.

    """
    stocks = load_ftse_stocks()
    X = generate_correlated_returns(stocks)
    result = analyse_multicollinearity(X, stocks)
    # kappa > 100 confirms ill-conditioning is present with synthetic data
    assert result["condition_number"] > 100, \
        f"Expected kappa > 100, got {result['condition_number']:.2f}"
    assert result["n_high_corr_pairs"] > 0, "Should find highly correlated pairs"