"""Step 8: EDA questions 11-15 (feature relationships and correlations). Charts go to reports/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data") / "model_data.csv"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

PPSF = "Price_per_SqFt_Rupees"
df = pd.read_csv(DATA_PATH)


def save_chart(file_name: str) -> None:
    """Save the current figure into reports/ and close it."""
    plt.tight_layout()
    plt.savefig(REPORT_DIR / file_name, dpi=120)
    plt.close()
    print("   chart saved:", REPORT_DIR / file_name)


# Q11: correlation between numeric features
numeric_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", PPSF, "Year_Built", "Floor_No",
                "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]
corr = df[numeric_cols].corr()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
pairs = upper.reindex(upper.abs().sort_values(ascending=False).index)
print("Q11 Five strongest correlations between numeric features:")
for (a, b), value in pairs.head(5).items():
    print(f"   {a} vs {b}: {value:.3f}")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Q11: Correlation Between Numeric Features")
save_chart("eda_q11_correlation_heatmap.png")


def relation_with_ppsf(column: str, question: str, file_name: str) -> None:
    """Print and plot how price per sq ft changes with a count column (schools or hospitals)."""
    mean_ppsf = df.groupby(column)[PPSF].mean()
    print(f"{question} Correlation of {column} with price per sq ft: {df[column].corr(df[PPSF]):.4f}")
    print(f"   Average price per sq ft (Rs) for {mean_ppsf.index.min()} to {mean_ppsf.index.max()}:",
          mean_ppsf.round(0).tolist())
    sns.lineplot(x=mean_ppsf.index, y=mean_ppsf.values, marker="o")
    plt.ylim(0, mean_ppsf.max() * 1.2)
    plt.title(f"{question} Average Price per Sq Ft by {column}")
    plt.xlabel(column)
    plt.ylabel("Average price per sq ft (Rs)")
    save_chart(file_name)


# Q12 and Q13: schools and hospitals vs price per sq ft
relation_with_ppsf("Nearby_Schools", "Q12", "eda_q12_schools_vs_ppsf.png")
relation_with_ppsf("Nearby_Hospitals", "Q13", "eda_q13_hospitals_vs_ppsf.png")

# Q14: price by furnished status
furnished = df.groupby("Furnished_Status").agg(
    mean_price=("Price_in_Lakhs", "mean"),
    median_price=("Price_in_Lakhs", "median"),
    mean_ppsf=(PPSF, "mean"),
).round(1)
print("Q14 Price by furnished status:")
print(furnished.to_string())
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=furnished.index, y=furnished["mean_price"], hue=furnished.index, legend=False, ax=axes[0])
axes[0].set_title("Average price (lakhs)")
sns.barplot(x=furnished.index, y=furnished["mean_ppsf"], hue=furnished.index, legend=False, ax=axes[1])
axes[1].set_title("Average price per sq ft (Rs)")
plt.suptitle("Q14: Price by Furnished Status")
save_chart("eda_q14_price_by_furnished.png")

# Q15: price per sq ft by facing direction
facing = df.groupby("Facing")[PPSF].agg(["mean", "median"]).round(0)
print("Q15 Price per sq ft (Rs) by facing direction:")
print(facing.to_string())
sns.barplot(x=facing.index, y=facing["median"], hue=facing.index, legend=False)
plt.title("Q15: Median Price per Sq Ft by Facing Direction")
plt.ylabel("Median price per sq ft (Rs)")
save_chart("eda_q15_ppsf_by_facing.png")