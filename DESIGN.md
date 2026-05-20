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
  Formula: Minimise ||Xw - y||^2 + lambda * ||w||^2 - (Ridge loss: fit to data + regularisation penalty)
  Solution: w = (X'X + lambda*I)^-1 X'y
  Constraints: 
  - Stock limit: 10%
  - Sector limit: 30%
  - Liquidity: 5% min
  - Normalisation: sum to 1.0 and non-negative
 
  ## 3. Architecture
  ```
  ftse100_portfolio_rebalancer/
  |-- src/
  |    |-- data.py 
  |    |         |- load_ftse_stocks(filepath)                                -> stocks, a list of stocks with ticker, sector, liquidity    
  |    |         |- generate_correlated_returns(stocks, n_days)               -> returns x matrix (252 x 100) where each cell is a daily return value    
  |    |         |_ analyse_multicollinearity(x)                              -> k value and highly correlated pairs    
  |    |
  |    |-- ridge.py  
  |    |         |- ridge_regression(x, y, lambda)                            -> portfolio weights w
  |    |         |- compute_condition_number(x, lambda)                       -> k value
  |    |         |- cross_validate_lambda(x, y)                               -> lambda
  |    |         |_ compare_conditioning(x, lambda)                           -> k before, k after, improvement factor
  |    |
  |    |-- constraints.py 
  |    |         |- normalise_weights(w)                                      -> normalised weights summing to 1.0
  |    |         |- project_weights(w, stocks,
  |    |         |                  max_stock=0.10, 
  |    |         |                  max_sector=0.30,
  |    |         |                  min_liquid=0.05,
  |    |         |                  max_iter=100,
  |    |         |                  tol=1e-6)                                 -> FCA compliant weights as np.ndarray
  |    |         |_ check_fca_compliance(w, stocks, tol=1e-4)                 -> dict {compliant: bool, violations: list}
  |    |
  |    |-- rebalance.py 
  |    |         |- generate_trade_list(current_weights,                      -> list of trade dicts {ticker, action, amount_gbp, weight_change}
  |    |         |                      target_weights,
  |    |         |                      stocks,
  |    |         |                      portfolio_value=50_000_000) 
  |    |         |_ estimate_transaction_costs(trades,                        -> dict {n_trades, fixed_costs, commission, stamp_duty, total_cost, cost_pct}
  |    |                                       fixed_cost=25.0,
  |    |                                       commission_rate=0.001)
  |    |__ analysis.py
  |              |- sector_allocation(w, stocks)                              -> sector weight dict
  |              |- compute_portfolio_volatility(w, returns)                  -> volatility score
  |              |_ compare_portfolios(current_w,                             -> dict {current_vol, optimised_vol, current_sector, optimised_sector}
  |                                    optimised_w,
  |                                    returns, 
  |                                    stocks)
  |-- data/
  |-- tests/
  |       |-- test_data.py
  |       |-- test_ridge.py
  |       |-- test_constraints.py
  |       |-- test_rebalance.py
  |       |__ test_integration.py
  |
  |-- results/                                                                -> stores CSV files created from export_results.py, stores png files created by 
  |                                                                              visualise_portfolio.py
  |-- images/                                                                 -> chart outputs for README display
  |-- main.py                                                                 -> orchestrates the full pipeline: load -> optimise -> constrain -> rebalance -> analyse
  |                                                                              -> export -> visualise
  |-- verify_multicollinearity.py                                             -> standalone diagnostic script. run once to confirm k(X'X) > 100 and ridge is required.
  |                                                                             
  |-- export_results.py 
  |              |- export_trades(trades, costs, output_dir)                  -> writes trades.csv
  |              |- export_compliance(compliance, output_dir)                 -> writes compliance.csv
  |              |- export_volatility(comparison, output_dir)                 -> writes volatility.csv
  |              |- export_sector_allocation(comparison, output_dir)          -> writes sector_allocation.csv
  |              |_ export_all(trades,                                        -> creates results/ and writes all 4 CSV files
  |                            costs,
  |                            compliance,
  |                            comparison,
  |                            output_dir)
  |                                                      
  |-- visualise_portfolio.py                                                  
  |              |- plot_sector_comparison(comparison, output_dir)            -> saves sector_comparison.png
  |              |- plot_sector_pie(comparison, output_dir)                   -> saves sector_pie.png
  |              |_ visualise_portfolio(comparison, output_dir)               -> creates results/ and save both charts
  |
  |-- DESIGN.md
  |-- README.md
  |-- .gitignore
  |__ requirements.txt

  ```

  ## 4. Risks 
  - X'X is still ill-conditioned after ridge regression - increase lambda
  - Constraint projection doesn't converge — cap iterations at 100, check weight change < 1e-6 between iterations
  - FCA constraint violation - check compliance after every optimisation, do not return portfolio without passing all
  - Overfitting to history - use lambda regularisation and out of sample test
 
  ## 5. Success Criteria 
  - All FCA constraints are satisfied
  - Weights sum to 1.0 and are non-negative
  - Portfolio weights are numerically stable
  - Report showing that condition number decreases with X'X + lambda*I
  - Trade list output showing current weight, target weight and trade direction per stock

  ## 6. Version 2 Roadmap

### Data Layer
- Replace `generate_correlated_returns()` in `src/data.py` with yfinance loader
- Fetch real FTSE 100 daily returns via `yfinance.download()`
- Expected condition number k(X'X) >> 10,000 with real data vs 2,862 synthetic
- Volatility comparison will become meaningful with real returns

### Solver Upgrade
- Replace iterative projection in `src/constraints.py` with constrained QP
- Use `scipy.optimize.minimize` with SLSQP or `cvxpy`
- FCA constraints become hard constraints — provably correct rather than approximated
- Convergence warning in current `project_weights()` will be eliminated

### Lambda Grid
- Expand cross-validation grid in `main.py` from [0.1, 1.0, 10.0]
- Use `np.logspace(-2, 3, 20)` for proper search across 20 values
- Affects `cross_validate_lambda()` in `src/ridge.py`

### y Target
- Replace cross-sectional mean `y = np.mean(returns, axis=1)` in `main.py`
- Use benchmark index return or forward-looking return estimate
- Document the chosen target and its financial justification

### Test Upgrades
- Add integration test with real data fixture
- Add QP solver correctness test against known analytical solution
- Expand lambda grid test to cover logspace range