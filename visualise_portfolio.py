"""
visualise_portfolio.py — Sector allocation charts.

"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_sector_comparison(comparison: dict, output_dir: str) -> None:
  """
  Grouped bar chart — sector weights before and after rebalance.
  
  """
  sectors = list(comparison["current_sector"].keys())
  current = [comparison["current_sector"][s] for s in sectors]
  optimised = [comparison["optimised_sector"][s] for s in sectors]

  x = np.arange(len(sectors))
  width = 0.35

  fig, ax = plt.subplots(figsize=(12, 6))
  ax.bar(x - width/2, current, width, label="Current", color="steelblue")
  ax.bar(x + width/2, optimised, width, label="Optimised", color="darkorange")
  ax.axhline(y=0.30, color="red", linestyle="--", label="FCA 30% limit")
  ax.set_xticks(x)
  ax.set_xticklabels(sectors, rotation=45, ha="right")
  ax.set_ylabel("Weight")
  ax.set_title("Sector Allocation — Before vs After Rebalance")
  ax.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "sector_comparison.png"))
  plt.close()

def plot_sector_pie(comparison: dict, output_dir: str) -> None:
  """
  Pie chart — optimised sector allocation.
  
  """
  sectors = list(comparison["optimised_sector"].keys())
  weights = [comparison["optimised_sector"][s] for s in sectors]

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.pie(weights, labels=sectors, autopct="%1.1f%%", startangle=90)
  ax.set_title("Optimised Portfolio — Sector Allocation")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "sector_pie.png"))
  plt.close()

def visualise_portfolio(comparison: dict, output_dir: str = "results") -> None:
  """
  Create output directory and save both charts.
  
  """
  os.makedirs(output_dir, exist_ok=True)
  plot_sector_comparison(comparison, output_dir)
  plot_sector_pie(comparison, output_dir)
  print(f"Charts saved to {output_dir}/")