import polars as pl

df = pl.read_csv("data/2024-citsci-pollutants.csv")
df.write_parquet("data/2024-citsci-pollutants.parquet")