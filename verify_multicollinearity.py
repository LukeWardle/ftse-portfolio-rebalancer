"""
verify_multicollinearity.py — Confirm Ridge is necessary before implementing constraints.

"""
from src.data import load_ftse_stocks, generate_correlated_returns
from src.data import analyse_multicollinearity
from src.ridge import cross_validate_lambda, compare_conditioning
import numpy as np

stocks = load_ftse_stocks()
X = generate_correlated_returns(stocks)
y = np.mean(X, axis=1)

print("=" * 62)
print("FTSE 100 Multicollinearity Diagnosis")
print("=" * 62)

analysis = analyse_multicollinearity(X, stocks)
print(f"\nk(X^t @ X) = {analysis['condition_number']:.2e}")
print(f"Mean pairwise correlation: {analysis['mean_pairwise_correlation']:.3f}")
print(f"High correlation pairs (|r|>0.75): {analysis['n_high_corr_pairs']}")

best_lam, cv_results = cross_validate_lambda(X, y)
print(f"\nCross-validation results (lambda -> avg MSE):")
for lam, mse in sorted(cv_results.items()):
    marker = "<- BEST" if lam == best_lam else ""
    print(f"lambda = {lam:6.2f} -> MSE = {mse:.6f}{marker}")

result = compare_conditioning(X, best_lam)
print(f"\nConditioning improvement (lambda = {best_lam}):")
print(f"k before ridge: {result['kappa_direct']:.2e}")
print(f"k after ridge:  {result['kappa_ridge']:.2e}")
print(f"Improvement:    {result['improvement_factor']:.0f}x")