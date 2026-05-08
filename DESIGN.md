# FTSE 100 Portfolio Rebalancing — Design Document 
  Author: Luke Wardle 
  Date: 07/05/2026 
  Status: Design Phase 
 
  ## 1. Requirements 
  Must-Have (FCA legal requirements): 
  - No single stock exceeds 10%
  - No sector exceeds 30%
  - Minimum 5% in liquid assets
  - All weights non-negative - no short selling
  - All weights sum to 1.0
  Must-Have (Technical): 
  - Ridge regression - used to combat ill-conditioned problem
  - Cross-validated lambda - hardcoded lambda is a data-dependent constant and not generalisable
  - Condition number reporting - must verify k value improvement
  - Trade list output - end user 
  - System must rebalance computation in under 5 minutes.
  Should-Have: 
  - Transaction cost minimisation - nice for business value not in acceptance criteria
  - Portfolio volatility reduction - not in acceptance criteria
  Out of Scope: 
  - Real-time data feeds
  - Backtesting
  - Visualisation and charts - good for portfolio however not in acceptance criteria
  - Transaction cost minimisation (full optimisation version)
  - Factor model decomposition
  - Multi-period optimisation
 
  ## 2. Problem Formulation 
  Type: Ridge regression with constraints 
  Objective: Minimise prediction error + penalise large weights
  Formula: Minimise ||Xw - y||^2 + lambda||w||^2 - (Ridge loss: fit to data + regularisation penalty)
  Solution: w = (X^T @ X + lambda * I)^-1 @ X^T @ y
  Constraints: 
  - Stock limit: 10%
  - Sector limit: 30%
  - Liquidity: 5% min
  - Normalisation: sum to 1.0 and non-negative
 
  ## 3. Architecture
  ```
  ftse100_portfolio_rebalancer/
  |-- src/
  |    |-- data.py # Loads FTSE stock metadata generates correlated returns matrix and analyses multicollinearity.
  |    |   |- load_ftse_stocks(filepath)                  -> stocks, a list of stocks with ticker, sector, liquidity    
  |    |   |- generate_correlated_returns(stocks, n_days) -> returns x matrix (252 x 100) where each cell is a daily return value    
  |    |   |_ analyse_multicollinearity(x)                -> k value and highly correlated pairs    
  |    |-- ridge.py — Ridge regression, calculate condition number, cross-validate lambda and compare conditioning 
  |    |   |- ridge_regression(x, y, lambda)              -> portfolio weights w
  |    |   |- compute_condition_number(x, lambda)         -> k value
  |    |   |- cross_validate_lambda(x, y)                 -> lambda
  |    |   |_ compare_conditioning(x, lambda)             -> k before, k after, improvement factor
  |    |-- constraints.py  — FCA compliance and constraints checker
  |    |   |- project_weights(w, constraints)             -> FCA compliant weights
  |    |   |_ check_fca_compliance(w, constraints)        -> bool (True = compliant, False = violation)
  |    |-- rebalance.py — Trade list generation 
  |    |   |_ generate_trade_list(current_weights, target_weights) -> trade list
  |    |-- analysis.py — Reporting - sector allocation, compute_portfolio_volatility - report the risk after rebalancing
  |        |- compute_sector_allocation(w, stocks)        -> sector weight dict
  |        |_ compute_portfolio_volatility(x, w)          -> volatility score
  |-- data/
  |-- tests/
  |    |-- test_data.py
  |    |-- test_ridge.py
  |    |-- test_constraints.py
  |    |-- test_rebalance.py
  |    |__ test_integration.py
  |-- results/
  |-- main.py - Orchestrates the full pipeline: load -> ridge -> FCA compliance -> validate -> output trade list
  |-- verify_multicollinearity.py - standalone diagnostic script. run once before implementation to confirm k(X^T @ x) > 10^6 and ridge is required.
  |-- DESIGN.md
  |-- README.md
  |-- .gitignore
  |__ requirements.txt

  ```

  ## 4. Risks 
  - X^t @ x is still ill-conditioned after ridge regression - increase lambda
  - Constraint projection doesn't converge — cap iterations at 100, check weight change < 1e-8 between iterations
  - FCA constraint violation - check compliance after every optimisation, do not return portfolio without passing all
  - Overfitting to history - use lambda regularisation and out of sample test
 
  ## 5. Success Criteria 
  - All FCA constraints are satisfied
  - Weights sum to 1.0 and are non-negative
  - Portfolio weights are numerically stable
  - Report showing that condition number decreases with X^t @ X + lambda * I
  - Trade list output showing current weight, target weight and trade direction per stock