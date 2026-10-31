"""Step 7: EDA questions 6-10 (location analysis). Charts are saved to reports/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write charts to files instead of opening windows
import matplotlib.pyplot as plt
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


def top_bottom(series: pd.Series, decimals: int = 1) -> None:
    """Print the three highest and three lowest values of a sorted series."""
    print("   highest 3:", series.head(3).round(decimals).to_dict())
    print("   lowest 3 :", series.tail(3).round(decimals).to_dict())
    print(f"   gap between highest and lowest: {(series.max() / series.min() - 1) * 100:.1f}%")


# Q6: average price per sq ft by state
by_state = df.groupby("State")[PPSF].mean().sort_values(ascending=False)
print(f"Q6 Average price per sq ft (Rs) by state ({len(by_state)} states):")
top_bottom(by_state, 0)
plt.figure(figsize=(9, 7))
sns.barplot(x=by_state.values, y=by_state.index, color="steelblue")
plt.title("Q6: Average Price per Sq Ft by State")
plt.xlabel("Average price per sq ft (Rs)")
save_chart("eda_q06_ppsf_by_state.png")

# Q7: average property price by city
by_city = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
print(f"Q7 Average price (lakhs) by city ({len(by_city)} cities):")
top_bottom(by_city, 1)
plt.figure(figsize=(9, 11))
sns.barplot(x=by_city.values, y=by_city.index, color="seagreen")
plt.title("Q7: Average Property Price by City")
plt.xlabel("Average price (lakhs)")
save_chart("eda_q07_price_by_city.png")

# Q8: median age of properties by locality
age_by_locality = df.groupby("Locality")["Age_of_Property"].median()
print(f"Q8 Median property age across {len(age_by_locality)} localities: "
      f"min {age_by_locality.min():.0f} | max {age_by_locality.max():.0f} "
      f"| overall median {df['Age_of_Property'].median():.0f}")
print("   Oldest 3 localities:", age_by_locality.sort_values(ascending=False).head(3).to_dict())
sns.histplot(age_by_locality, discrete=True, color="darkorange")
plt.title("Q8: Median Property Age by Locality (one value per locality)")
plt.xlabel("Median age (years)")
save_chart("eda_q08_age_by_locality.png")

# Q9: BHK distribution across cities
bhk_share = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
print("Q9 Overall BHK share (%):", (df["BHK"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print(f"   Share of a BHK type within a city ranges from {bhk_share.min().min():.1f}% to {bhk_share.max().max():.1f}%")
plt.figure(figsize=(7, 14))
sns.heatmap(bhk_share, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "% of city's properties"})
plt.title("Q9: BHK Distribution Across Cities (%)")
save_chart("eda_q09_bhk_by_city.png")

# Q10: price trend for the top 5 most expensive localities
# The dataset has no sale dates, so the only time dimension is the year the property was built.
locality_stats = df.groupby("Locality")["Price_in_Lakhs"].agg(["mean", "count"]).sort_values("mean", ascending=False)
top5 = locality_stats.head(5)
print("Q10 Top 5 localities by average price (lakhs):")
print(top5.round(1).to_string())
bins = [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023]
labels = ["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23"]
subset = df[df["Locality"].isin(top5.index)].copy()
subset["Built_Period"] = pd.cut(subset["Year_Built"], bins=bins, labels=labels)
trend = subset.groupby(["Locality", "Built_Period"], observed=True)["Price_in_Lakhs"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=trend, x="Built_Period", y="Price_in_Lakhs", hue="Locality", marker="o")
plt.title("Q10: Average Price by Construction Period, Top 5 Localities")
plt.xlabel("Year built")
plt.ylabel("Average price (lakhs)")
save_chart("eda_q10_top5_locality_trend.png")