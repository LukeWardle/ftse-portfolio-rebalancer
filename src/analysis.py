"""
analysis.py - Portfolio risk metrics and comparison.

"""
import numpy as np
from typing import List, Dict

def compute_portfolio_volatility(weights: np.ndarray,
                                 returns: np.ndarray) -> float:
  """
  Annualised portfolio volatility: sigma = sqrt(w^t @ cov @ weights * 252)

  """
  cov = np.cov(returns, rowvar=False)
  portfolio_var = float(weights.T @cov @ weights)
  return float(np.sqrt(portfolio_var * 252))

def sector_allocation(weights: np.ndarray,
                      stocks: List[Dict]) -> Dict[str, float]:
  """
  Return dict mapping sector name -> total weight.
  
  """
  alloc = {}
  for i, s in enumerate(stocks):
    alloc[s["sector"]] = alloc.get(s["sector"], 0) + float(weights[i])
  return alloc

def compare_portfolios(current_w, optimised_w, returns, stocks) -> Dict:
  """
  Report vol and sector change for current vs optimised.
  
  """
  return {
    "current_vol":      compute_portfolio_volatility(current_w, returns),
    "optimised_vol":    compute_portfolio_volatility(optimised_w, returns),
    "current_sector":   sector_allocation(current_w, stocks),
    "optimised_sector": sector_allocation(optimised_w, stocks),
  }