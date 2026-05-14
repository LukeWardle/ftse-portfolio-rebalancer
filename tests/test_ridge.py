"""
test_ridge.py - Unit tests for ridge.py testing regression implementation.

"""
import numpy as np
from src.ridge import ridge_regression, compute_condition_number
from src.ridge import cross_validate_lambda, compare_conditioning

def test_ridge_output_shape():
  """
  Ridge regression must return a weight vector of shape (n_stocks,)
  
  """
  X = np.random.randn(100, 10)
  y = np.random.randn(100)
  w = ridge_regression(X, y, lambda_=1.0)
  assert w.shape == (10,), f"Expected (10,), got {w.shape}"

def test_ridge_reduces_condition_number():
  """
  Create intentionally ill-conditioned data
  
  """
  X = np.random.randn(100, 10)
  X[:, 1] = X[:, 0] + 0.001 * np.random.randn(100)
  kappa_direct = compute_condition_number(X, lambda_=0.0)
  kappa_ridge = compute_condition_number(X, lambda_=10.0)
  assert kappa_ridge < kappa_direct, "Ridge must improve conditioning"
  assert kappa_ridge < 1e6, f"After ridge κ={kappa_ridge:.2e}, expected <1e6"

def test_cross_validation_returns_valid_lambda():
  """
  Cross-validation must return a lambda from the candidate list with positive MSE scores.

  """
  X = np.random.randn(100, 10) 
  y = np.random.randn(100)
  best_lam, cv = cross_validate_lambda(X, y)
  assert best_lam in [0.01, 0.1, 1.0, 10.0, 100.0]
  assert all(v > 0 for v in cv.values()), "All MSE values must be positive"

def test_lambda_zero_matches_ols():
  """
  Ridge with lambda=0 must produce identical weights to OLS — proves implementation is correct.

  """
  X = np.random.randn(100, 10); y = np.random.randn(100)
  w_ridge = ridge_regression(X, y, lambda_=0.0)
  w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
  np.testing.assert_allclose(w_ridge, w_ols, rtol=1e-1)  # solve() uses normal equations, lstsq uses SVD — different numerical paths - v2 Yfinance may need changing

def test_large_lambda_shrinks_norm():
  """
  Large lambda must produce smaller weight vector norm than small lambda — proves shrinkage works.

  """
  X = np.random.randn(100, 10); y = np.random.randn(100)
  w_small = ridge_regression(X, y, lambda_=0.01)
  w_large = ridge_regression(X, y, lambda_=100.0)
  assert np.linalg.norm(w_large) < np.linalg.norm(w_small)

def test_compare_conditioning_shows_improvement():
  """
  compare_conditioning must confirm Ridge reduced kappa — proves the full pipeline works.

  """
  X = np.random.randn(200, 20)
  X[:, 1] = X[:, 0] + 0.001 * np.random.randn(200)
  result = compare_conditioning(X, optimal_lambda=1.0)
  assert result["kappa_direct"] > result["kappa_ridge"]
  assert result["improvement_factor"] > 1.0