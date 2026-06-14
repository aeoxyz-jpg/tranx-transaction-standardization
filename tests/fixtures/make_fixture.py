"""Generate a tiny fixture parquet that mimics the HF dataset schema."""
from pathlib import Path
import polars as pl

rows = []
data = [
    ("McDonald's #111", "Food & Dining", "USA", "USD"),
    ("McDonald's #121", "Food & Dining", "USA", "USD"),
    ("BP on Buford Hwy", "Transportation", "USA", "USD"),
    ("BP @ Pleasant Hills", "Transportation", "USA", "USD"),
    ("Amazon - AUSTRALIA", "Shopping & Retail", "AUSTRALIA", "AUD"),
    ("Walgreens #8780", "Healthcare & Medical", "UK", "GBP"),
    ("Wage", "Income", "USA", "USD"),
    ("Salary - Rush Hour", "Income", "CANADA", "CAD"),
    ("Starbucks - AUSTRALIA Store", "Food & Dining", "AUSTRALIA", "AUD"),
    ("Disney+ #2115", "Entertainment & Recreation", "INDIA", "INR"),
]
# Repeat to give stratified sampling something to work with
for i in range(300):
    desc, cat, country, cur = data[i % len(data)]
    rows.append({"transaction_description": desc, "category": cat,
                 "country": country, "currency": cur})

df = pl.DataFrame(rows)
out = Path(__file__).parent / "sample.parquet"
df.write_parquet(out)
print(f"wrote {out} ({len(df)} rows)")
