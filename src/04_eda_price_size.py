"""Step 6: EDA questions 1-5 (price and size analysis). Charts are saved to reports/."""
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


def iqr_outliers(series: pd.Series) -> int:
    """Count values outside 1.5 x IQR of the quartiles."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


# Q1: distribution of property prices
price = df["Price_in_Lakhs"]
print(f"Q1 Price (lakhs): mean {price.mean():.1f} | median {price.median():.1f} | skew {price.skew():.3f}")
sns.histplot(price, bins=25, color="steelblue")
plt.title("Q1: Distribution of Property Prices")
plt.xlabel("Price (lakhs)")
save_chart("eda_q01_price_distribution.png")

# Q2: distribution of property sizes
size = df["Size_in_SqFt"]
print(f"Q2 Size (sq ft): mean {size.mean():.0f} | median {size.median():.0f} | skew {size.skew():.3f}")
sns.histplot(size, bins=25, color="seagreen")
plt.title("Q2: Distribution of Property Sizes")
plt.xlabel("Size (sq ft)")
save_chart("eda_q02_size_distribution.png")

# Q3: price per sq ft by property type
print("Q3 Median price per sq ft (Rs) by property type:")
print(df.groupby("Property_Type")[PPSF].median().round(0).to_string())
sns.boxplot(data=df, x="Property_Type", y=PPSF, hue="Property_Type", legend=False)
plt.yscale("log")
plt.title("Q3: Price per Sq Ft by Property Type (log scale)")
plt.ylabel("Price per sq ft (Rs)")
save_chart("eda_q03_ppsf_by_type.png")

# Q4: relationship between size and price
print(f"Q4 Correlation between size and price: {size.corr(price):.4f}")
quintiles = pd.qcut(size, 5)
print("   Median price (lakhs) in each size quintile:",
      df.groupby(quintiles, observed=True)["Price_in_Lakhs"].median().round(1).tolist())
sample = df.sample(5000, random_state=42)
sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3, s=12)
slope, intercept = np.polyfit(sample["Size_in_SqFt"], sample["Price_in_Lakhs"], 1)
xs = np.array([500, 5000])
plt.plot(xs, slope * xs + intercept, color="red", label="trend line")
plt.legend()
plt.title("Q4: Property Size vs Price (5,000 random properties)")
save_chart("eda_q04_size_vs_price.png")

# Q5: outliers in price per sq ft and property size
for column in ["Size_in_SqFt", "Price_in_Lakhs", PPSF]:
    count = iqr_outliers(df[column])
    print(f"Q5 IQR outliers in {column}: {count} ({count / len(df) * 100:.1f}%)")
q1, q3 = df[PPSF].quantile([0.25, 0.75])
high = df[df[PPSF] > q3 + 1.5 * (q3 - q1)]
print(f"   High price-per-sq-ft outliers: average size {high['Size_in_SqFt'].mean():.0f} sq ft "
      f"vs {size.mean():.0f} for all properties")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, column in zip(axes, ["Size_in_SqFt", "Price_in_Lakhs", PPSF]):
    sns.boxplot(y=df[column], ax=ax, color="lightcoral")
    ax.set_title(column)
axes[2].set_yscale("log")
plt.suptitle("Q5: Outliers in Size, Price and Price per Sq Ft")
save_chart("eda_q05_outliers.png")