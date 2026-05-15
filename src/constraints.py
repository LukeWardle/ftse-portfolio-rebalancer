"""
constraints.py - FCA regulatory constraint projection

Projects unconstrained Ridge weights onto the FCA feasible set:
- w_j >= 0          # No short selling - weights must be positive
- sum(w) = 1.0      # Entire portfolio must be fully invested
- w_j <= 0.10       # Max 10% in any given stock 
- sector <= 0.30    # Max 30% in any given sector
- liquid >= 0.05    # Min 5% in Liquidity stocks

"""
import numpy as np
from typing import List, Dict

def normalise_weights(w: np.ndarray) -> np.ndarray:
  """
  Normalise w so all elements sum to 1.0

  """
  total = w.sum()
  if abs(total) < 1e-10:
    return np.ones_like(w) / len(w) # edge: all-zero -> equal weight
  return w / total

def project_weights(w: np.ndarray,
                    stocks: List[Dict],
                    max_stock: float = 0.10,
                    max_sector: float = 0.30,
                    min_liquid: float = 0.05,
                    max_iter: int = 100,
                    tol: float = 1e-6) -> np.ndarray:
  """
  Project w onto FCA constraint set via iterative projection.
  Converges when ||w_new - w_old|| < tol (typically 3-15 iterations).
  
  """
  w_proj = w.copy()

  # Pre-compute sector and liquidity indices once
  sectors = {}
  for i, s in enumerate(stocks):
    sectors.setdefault(s["sector"], []).append(i)
  liquid_idx = [i for i, s in enumerate(stocks) if s["liquid"]]
  illiquid_idx = [i for i in range(len(stocks)) if i not in liquid_idx]

  for _ in range(max_iter):
    w_old = w_proj.copy()

    # Step 1: Remove negatives
    w_proj = np.maximum(w_proj, 0.0)

    # Step 2: Normalise to sum = 1
    w_proj = normalise_weights(w_proj)

    # Step 3: Clip individual stocks to max_stock (10%)
    w_proj = np.minimum(w_proj, max_stock)
    w_proj = normalise_weights(w_proj)

    # Step 4: Enforce sector limits (30%)
    for sec_idx in sectors.values():
      sec_wt = w_proj[sec_idx].sum()
      if sec_wt > max_sector:
        w_proj[sec_idx] *= max_sector / sec_wt
    w_proj = normalise_weights(w_proj)

    # Step 5: Enforce liquidity minimum (5%)
    liq_wt = w_proj[liquid_idx].sum()
    if liq_wt < min_liquid and illiquid_idx:
      deficit = min_liquid - liq_wt
      reduce = deficit / len(illiquid_idx)
      w_proj[illiquid_idx] = np.maximum(w_proj[illiquid_idx] - reduce, 0.0)
      w_proj[liquid_idx] += deficit / len(liquid_idx)
    w_proj = normalise_weights(w_proj)

    # Convergence check
    if np.linalg.norm(w_proj - w_old) < tol:
      break

  return w_proj

def check_fca_compliance(w: np.ndarray,
                         stocks: List[Dict],
                         tol: float = 1e-4) -> Dict:
  """
  Validate all FCA constraints. Call after every optimisation.
  Returns dict with compliant flag and list of violation strings.
  
  """
  violations = []

  # Rule 1: weights sum to 1.0
  if abs(w.sum() - 1.0) > tol:
    violations.append(f"Sum = {w.sum():.4f}, expected 1.0")

  # Rule 2: no negative weights
  neg = np.where(w < -tol)[0]
  if len(neg):
    violations.append(f"{len(neg)} negative weights")

  # Rule 3: no single stock above 10%
  over = np.where(w > 0.10 + tol)[0]
  if len(over):
    violations.append(f"{len(over)} stocks exceed 10% limit")

  # Rule 4: no sector above 30%
  sectors = {}
  for i, s in enumerate(stocks):
    sectors.setdefault(s["sector"], []).append(i)
  for sec, idx in sectors.items():
    sec_wt = w[idx].sum()
    if sec_wt > 0.30 + tol:
      violations.append(f"Sector {sec}: {sec_wt:.1%} > 30%")
  
  # Rule 5: liquid assets at least 5%
  liq_wt = w[[i for i, s in enumerate(stocks) if s["liquid"]]].sum()
  if liq_wt < 0.05 - tol:
    violations.append(f"Liquid: {liq_wt:.1%} < 5%")

  return {"compliant": not violations, "violations": violations}