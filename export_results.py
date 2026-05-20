"""
export_results.py - Export pipeline results to CSV files.

"""
import os
import csv
from typing import List, Dict

def export_trades(trades: List[Dict], costs: Dict, output_dir: str) -> None:
  """
  Write trade list and cost summary to trades.csv.
  
  """
  filepath = os.path.join(output_dir, "trades.csv")
  with open(filepath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ticker", "action", "amount_gbp", "weight_change"])
    writer.writeheader()
    writer.writerows(trades)

def export_compliance(compliance: Dict, output_dir: str) -> None:
  """
  Write FCA compliance result to compliance.csv.
  
  """
  filepath = os.path.join(output_dir, "compliance.csv")
  with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["compliant", "violations"])
    writer.writerow([compliance["compliant"], "; ".join(compliance["violations"])])

def export_volatility(comparison: Dict, output_dir: str) -> None:
  """
  Write before/after volatility to volatility.csv.
  
  """
  filepath = os.path.join(output_dir, "volatility.csv")
  with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "current", "optimised"])
    writer.writerow(["volatility", comparison["current_vol"], comparison["optimised_vol"]])

def export_sector_allocation(comparison: Dict, output_dir: str) -> None:
  """
  Write before/after sector allocation to sector_allocation.csv.
  
  """
  filepath = os.path.join(output_dir, "sector_allocation.csv")
  with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sector", "current", "optimised"])
    for sector in comparison["current_sector"]:
      writer.writerow([
        sector,
        comparison["current_sector"][sector],
        comparison["optimised_sector"][sector]
      ])

def export_all(trades: List[Dict], costs: Dict, compliance: Dict,
               comparison: Dict, output_dir: str = "results") -> None:
  """
  Create output directory and export all four CSV files.
  
  """
  os.makedirs(output_dir, exist_ok=True)
  export_trades(trades, costs, output_dir)
  export_compliance(compliance, output_dir)
  export_volatility(comparison, output_dir)
  export_sector_allocation(comparison, output_dir)
  print(f"Results exported to {output_dir}/")
    