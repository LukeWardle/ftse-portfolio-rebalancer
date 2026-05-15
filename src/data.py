"""
data.py - FTSE 100 data loading and multicollinearity analysis

"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def load_ftse_stocks(filepath: str = "data/ftse100.json") -> List[Dict]:
  """
  Load FTSE 100 stock metadata - ticker, name, sector, liquid flag.

  """
  with open(Path(filepath)) as f:
    stocks = json.load(f)
  required = {"ticker", "name", "sector", "liquid"}
  for s in stocks:
    if not required.issubset(s.keys()):
      raise ValueError(f"Missing fields in stock: {s}")
  return stocks

def generate_correlated_returns(stocks: List[Dict],
                                n_days: int = 252,
                                seed: int = 42) -> np.ndarray:
  """
  Generate synthetic FTSE-like returns with sector correlation.
  Model: return_i = 0.7 * sector_factor + 0.3 * idiosyncratic_noise
  This produces within-sector correlation of approxmately 0.70-0.85.
  
  """
  rng = np.random.default_rng(seed)
  n = len(stocks)
  unique_sectors = list(dict.fromkeys(s["sector"] for s in stocks))
  sector_factors = {sec: rng.standard_normal(n_days) * 0.015
  for sec in unique_sectors}
  X = np.zeros((n_days, n))
  for j, stock in enumerate(stocks):
    sector_noise = sector_factors[stock["sector"]]
    idio_noise = rng.standard_normal(n_days) * 0.008
    X[:, j] = 0.7 * sector_noise + 0.3 * idio_noise
  return X

def analyse_multicollinearity(X: np.ndarray, stocks: List[Dict]) -> Dict:
  """
  Compute condition number of X^t @ X and identify highly correlated pairs.
  Returns diagnostics dict - use to confirm Ridge is necessary.
  
  """
  XtX = X.T @ X
  kappa = float(np.linalg.cond(XtX))
  corr = np.corrcoef(X, rowvar=False)
  n = len(stocks)
  high_corr = []
  for i in range(n):
    for j in range(i + 1, n):
      if abs(corr[i, j]) > 0.75:
        high_corr.append({
          "stock1": stocks[i]["ticker"],
          "stock2": stocks[j]["ticker"],
          "r": round(float(corr[i, j]), 3),
          "same_sector": stocks[i]["sector"] == stocks[j]["sector"]
        })
  upper = corr[np.triu_indices(n, k=1)]
  return {
    "condition_number": kappa,
    "high_correlation": high_corr,
    "mean_pairwise_correlation": float(upper.mean()),
    "n_high_corr_pairs": len(high_corr)
  }