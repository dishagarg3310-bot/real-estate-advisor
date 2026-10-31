"""Step 4: clean the housing dataset and save a cleaned CSV (no rows are removed)."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data") / "india_housing_prices.csv"
CLEAN_PATH = Path("data") / "cleaned_housing.csv"

TEXT_COLS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw housing data."""
    df = df.copy()

    # 1. Remove duplicate records (none expected) and trim stray spaces in text columns
    df = df.drop_duplicates(subset="ID")
    for col in TEXT_COLS:
        df[col] = df[col].str.strip()

    # 2. Exact price per sq ft in rupees (the stored column is rounded to 2 decimals, in lakhs)
    df["Price_per_SqFt_Rupees"] = df["Price_in_Lakhs"] * 100_000 / df["Size_in_SqFt"]

    # 3. Flag impossible floor data instead of deleting almost half of the rows
    df["Floor_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)

    # 4. Turn the Amenities text into a count plus one 0/1 column per amenity
    amenity_lists = df["Amenities"].str.split(",").apply(lambda items: [a.strip() for a in items])
    df["Amenity_Count"] = amenity_lists.str.len()
    for amenity in sorted({a for items in amenity_lists for a in items}):
        df[f"Amenity_{amenity}"] = amenity_lists.apply(lambda items, a=amenity: int(a in items))
    return df


def iqr_outliers(series: pd.Series) -> int:
    """Count values outside 1.5 x IQR of the quartiles."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean(raw)

    print("Rows before / after:", len(raw), "/", len(cleaned))
    print("New columns:", [c for c in cleaned.columns if c not in raw.columns])
    print("Floor_Inconsistent rows:", int(cleaned["Floor_Inconsistent"].sum()))
    for col in ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rupees"]:
        print(f"IQR outliers in {col}:", iqr_outliers(cleaned[col]))
    print("Amenity_Count distribution:", cleaned["Amenity_Count"].value_counts().sort_index().to_dict())

    cleaned.to_csv(CLEAN_PATH, index=False)
    print("Saved", CLEAN_PATH, cleaned.shape)