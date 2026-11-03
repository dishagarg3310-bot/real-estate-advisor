"""Feature definitions and preprocessing shared by model training and the Streamlit app.

Run this file directly (python src/features.py) to check the feature setup.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data") / "model_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASSIFICATION_TARGET = "Good_Investment"
REGRESSION_TARGET = "Future_Price_5Y"

CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status",
]
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rupees", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Inconsistent", "Nearby_Schools", "Nearby_Hospitals",
    "Infrastructure_Score", "Amenity_Count", "Amenity_Clubhouse", "Amenity_Garden",
    "Amenity_Gym", "Amenity_Playground", "Amenity_Pool",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that must never be model inputs:
# - the two targets and the columns that directly define them (Assumed_Growth_Rate,
#   Investment_Score, City_Median_Price_per_SqFt)
# - ID (no information), Locality (500 labels with no signal in the EDA)
# - Year_Built, Price_per_SqFt, Amenities (duplicates of columns we already use in a better form)
EXCLUDED_COLUMNS = [
    "ID", "Locality", "Year_Built", "Price_per_SqFt", "Amenities", "Assumed_Growth_Rate",
    "City_Median_Price_per_SqFt", "Investment_Score", CLASSIFICATION_TARGET, REGRESSION_TARGET,
]

TRANSPORT_SCORE = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


def compute_infrastructure_score(df: pd.DataFrame) -> pd.Series:
    """Infrastructure score from 0 to 1 (schools 40%, hospitals 30%, public transport 30%)."""
    schools = (df["Nearby_Schools"] - 1) / 9
    hospitals = (df["Nearby_Hospitals"] - 1) / 9
    transport = df["Public_Transport_Accessibility"].map(TRANSPORT_SCORE)
    return 0.4 * schools + 0.3 * hospitals + 0.3 * transport


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical columns and scale the numeric columns."""
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])


def load_data() -> pd.DataFrame:
    """Load the modelling dataset created by src/03_create_targets.py."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 train-test split, stratified on the classification target."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[CLASSIFICATION_TARGET]
    )


if __name__ == "__main__":
    data = load_data()
    train_df, test_df = split_data(data)

    print("Rows: train", len(train_df), "| test", len(test_df))
    print("Good_Investment share: train", round(train_df[CLASSIFICATION_TARGET].mean() * 100, 1),
          "% | test", round(test_df[CLASSIFICATION_TARGET].mean() * 100, 1), "%")
    print("Features:", len(CATEGORICAL_FEATURES), "categorical +", len(NUMERIC_FEATURES), "numeric =", len(FEATURES))

    assert not set(FEATURES) & set(EXCLUDED_COLUMNS), "A leaking column is used as a feature!"
    print("Leakage check passed: no excluded column is a feature")
    print("Missing values in features:", int(data[FEATURES].isnull().sum().sum()))

    difference = (compute_infrastructure_score(data) - data["Infrastructure_Score"]).abs().max()
    print("Infrastructure score matches the stored column (max difference):", round(float(difference), 10))

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train_df[FEATURES])
    x_test = preprocessor.transform(test_df[FEATURES])
    print("Preprocessed shape: train", x_train.shape, "| test", x_test.shape)