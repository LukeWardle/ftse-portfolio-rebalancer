"""
generate_ftse_stocks.py - A module for generating synthetic data 

"""

import json
from pathlib import Path

sectors = [
  ("Financials", 25),
  ("Energy", 20),
  ("Consumer", 20),
  ("Healthcare", 20),
  ("Industrials", 15)
]
stocks= []
abbreviation = {"Financials": "FIN","Energy": "ENE","Consumer": "CON", "Healthcare": "HECA","Industrials": "IND"}
stock_index = 0
for sector_name, count in sectors:
  for i in range(count):
    ticker = abbreviation[sector_name] + str(i+1)
    name = sector_name + " stock" + str(i+1)
    sector = sector_name
    liquid = stock_index < 30
    stocks.append({
      "ticker": ticker,
      "name": name,
      "sector": sector,
      "liquid": liquid,
    })
    stock_index += 1

output_path = Path("data/ftse100.json")
with open(output_path, "w") as f:
  json.dump(stocks, f, indent=2)