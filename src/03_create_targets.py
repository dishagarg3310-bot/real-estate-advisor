"""Step 5: create the two rule-based target columns and save the modelling dataset."""
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path("data") / "cleaned_housing.csv"
MODEL_PATH = Path("data") / "model_data.csv"

BASE_GROWTH = 0.05          # minimum yearly price growth
MAX_EXTRA_GROWTH = 0.05     # extra yearly growth for the best infrastructure
YEARS = 5
GOOD_SCORE_THRESHOLD = 3    # Investment_Score >= 3 means "Good Investment"

TRANSPORT_SCORE = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the infrastructure score and both target columns."""
    df = df.copy()

    # Infrastructure score from 0 (poor) to 1 (excellent)
    schools = (df["Nearby_Schools"] - 1) / 9
    hospitals = (df["Nearby_Hospitals"] - 1) / 9
    transport = df["Public_Transport_Accessibility"].map(TRANSPORT_SCORE)
    df["Infrastructure_Score"] = 0.4 * schools + 0.3 * hospitals + 0.3 * transport

    # Regression target: price after 5 years
    df["Assumed_Growth_Rate"] = BASE_GROWTH + MAX_EXTRA_GROWTH * df["Infrastructure_Score"]
    df["Future_Price_5Y"] = (df["Price_in_Lakhs"] * (1 + df["Assumed_Growth_Rate"]) ** YEARS).round(2)

    # Classification target: multi-factor investment score
    city_median = df.groupby("City")["Price_per_SqFt_Rupees"].transform("median")
    df["City_Median_Price_per_SqFt"] = city_median.round(2)
    df["Investment_Score"] = (
        2 * (df["Price_per_SqFt_Rupees"] <= city_median).astype(int)
        + (df["BHK"] >= 3).astype(int)
        + (df["Availability_Status"] == "Ready_to_Move").astype(int)
        + (df["Public_Transport_Accessibility"] == "High").astype(int)
    )
    df["Good_Investment"] = (df["Investment_Score"] >= GOOD_SCORE_THRESHOLD).astype(int)
    return df


if __name__ == "__main__":
    data = add_targets(pd.read_csv(CLEAN_PATH))

    print("Shape:", data.shape)
    print("Infrastructure_Score mean:", round(data["Infrastructure_Score"].mean(), 3))
    print("Growth rate: mean", round(data["Assumed_Growth_Rate"].mean(), 4),
          "| min", round(data["Assumed_Growth_Rate"].min(), 4),
          "| max", round(data["Assumed_Growth_Rate"].max(), 4))
    ratio = data["Future_Price_5Y"] / data["Price_in_Lakhs"]
    print("Future / current price ratio: mean", round(ratio.mean(), 3))
    print("Future_Price_5Y (lakhs): median", round(data["Future_Price_5Y"].median(), 1),
          "| max", round(data["Future_Price_5Y"].max(), 1))
    print("Investment_Score counts:", data["Investment_Score"].value_counts().sort_index().to_dict())
    print("Good_Investment share:", round(data["Good_Investment"].mean() * 100, 1), "%")

    data.to_csv(MODEL_PATH, index=False)
    print("Saved", MODEL_PATH)