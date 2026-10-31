"""Step 3: first look at the housing dataset (read-only checks)."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data") / "india_housing_prices.csv"
df = pd.read_csv(DATA_PATH)

print("Shape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes.to_string())
print("\nMissing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Duplicate IDs:", int(df["ID"].duplicated().sum()))

print("\nNumeric summary:")
print(df.describe().T[["min", "50%", "max"]].round(2).to_string())

print("\nRows where Floor_No > Total_Floors:", int((df["Floor_No"] > df["Total_Floors"]).sum()))
print("Median Price_per_SqFt in the file:", round(df["Price_per_SqFt"].median(), 3))
print("Median Price_in_Lakhs / Size_in_SqFt:", round((df["Price_in_Lakhs"] / df["Size_in_SqFt"]).median(), 3))
print("Age_of_Property equals 2025 - Year_Built in",
      round(((2025 - df["Year_Built"]) == df["Age_of_Property"]).mean() * 100, 1), "% of rows")