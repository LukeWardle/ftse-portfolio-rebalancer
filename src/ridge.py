"""
ridge.py - Ridge regression for ill-conditioned portfolio optimisation
Implements w = (X^t @ X + lambda I) -1 X^ty from first principles.

"""
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional

def ridge_regression(X: np.ndarray,
                     y: np.ndarray,
                     lambda_: float) -> np.ndarray:
  """
  Compute Ridge regression weights: w = (X^t @ X + lambda I)-1 X^ty
  Args:
    X: Returns matrix (n_daya, n_stocks)
    y: Target returns (n_days,)
    lambda_: Regularisation parameter (>= 0)
  Returns:
    w: Portfolio weights (n_stocks,) - unconstrained
  
  """
  if lambda_ < 0:
    raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
  n_stocks = X.shape[1]
  XtX = X.T @ X
  Xty = X.T @ y
  ridge_matrix = XtX + lambda_ * np.eye(n_stocks)
  w = np.linalg.solve(ridge_matrix, Xty)
  return w

def compute_condition_number(X: np.ndarray,
                             lambda_: float = 0.0) -> float:
  """
  Compute k(X^t @ X + lambda I). Call with lambda_=0 for raw X^t @ X conditioning.
  Use before and after ridge to measure the improvement.
  
  """
  n = X.shape[1]
  matrix = X.T @ X + lambda_ * np.eye(n)
  return float(np.linalg.cond(matrix))

def cross_validate_lambda(X: np.ndarray,
                          y: np.ndarray,
                          lambdas: Optional[List[float]] = None,
                          k_folds: int = 5) -> Tuple[float, Dict[float, float]]:
  """
  Find optimal lambda by k-fold cross-validation.
  Returns:
    best_lambda: float - the lambda with lowest average CV MSE
    avg_mse: dict mapping each lambda -> average CV MSE
  
  """
  if lambdas is None:
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
  kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
  fold_mses = {lam: [] for lam in lambdas}
  for lam in lambdas:
    for train_idx, val_idx in kf.split(X):
      X_tr, X_val = X[train_idx], X[val_idx]
      y_tr, y_val = y[train_idx], y[val_idx]
      w = ridge_regression(X_tr, y_tr, lam)
      y_pred = X_val @ w
      mse = float(np.mean((y_val - y_pred) ** 2))
      fold_mses[lam].append(mse)
  avg_mse = {lam:float(np.mean(fold_mses[lam])) for lam in lambdas}
  best_lambda = min(avg_mse, key=avg_mse.get)
  return best_lambda, avg_mse

def compare_conditioning(X: np.ndarray,
                         optimal_lambda: float) -> Dict:
  """
  Report κ before and after ridge regularisation.
  This is the quantitative proof that Ridge did its job.
  
  """
  kappa_direct = compute_condition_number(X, lambda_=0.0)
  kappa_ridge = compute_condition_number(X, lambda_=optimal_lambda)
  return {
    "kappa_direct": kappa_direct,
    "kappa_ridge": kappa_ridge,
    "improvement_factor": kappa_direct / kappa_ridge,
    "lambda_used": optimal_lambda,
  }