"""Step 9: EDA questions 16-20 (ownership, availability, parking, amenities, transport)."""
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


def count_chart(column: str, title: str, file_name: str) -> None:
    """Bar chart with the number of properties in each category."""
    counts = df[column].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel("Number of properties")
    save_chart(file_name)


# Q16: properties per owner type
print("Q16 Properties per owner type:", df["Owner_Type"].value_counts().to_dict())
count_chart("Owner_Type", "Q16: Properties by Owner Type", "eda_q16_owner_type.png")

# Q17: properties per availability status
print("Q17 Properties per availability status:", df["Availability_Status"].value_counts().to_dict())
count_chart("Availability_Status", "Q17: Properties by Availability Status", "eda_q17_availability.png")

# Q18: does parking space affect price?
parking = df.groupby("Parking_Space").agg(
    properties=("ID", "count"),
    mean_price=("Price_in_Lakhs", "mean"),
    median_price=("Price_in_Lakhs", "median"),
    mean_ppsf=(PPSF, "mean"),
).round(1)
print("Q18 Price by parking space:")
print(parking.to_string())
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(x=parking.index, y=parking["mean_price"], hue=parking.index, legend=False, ax=axes[0])
axes[0].set_title("Average price (lakhs)")
sns.barplot(x=parking.index, y=parking["mean_ppsf"], hue=parking.index, legend=False, ax=axes[1])
axes[1].set_title("Average price per sq ft (Rs)")
plt.suptitle("Q18: Price With and Without Parking")
save_chart("eda_q18_parking.png")

# Q19: do amenities affect price per sq ft?
amenity_cols = [c for c in df.columns if c.startswith("Amenity_") and c != "Amenity_Count"]
by_count = df.groupby("Amenity_Count")[PPSF].mean()
print(f"Q19 Correlation of amenity count with price per sq ft: {df['Amenity_Count'].corr(df[PPSF]):.4f}")
print("   Average price per sq ft (Rs) for 1 to 5 amenities:", by_count.round(0).tolist())
differences = {}
for col in amenity_cols:
    has = df[df[col] == 1][PPSF].mean()
    lacks = df[df[col] == 0][PPSF].mean()
    differences[col.replace("Amenity_", "")] = has - lacks
    print(f"   {col.replace('Amenity_', '')}: with {has:.0f} | without {lacks:.0f} | difference {has - lacks:+.0f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=by_count.index, y=by_count.values, color="steelblue", ax=axes[0])
axes[0].set_title("Average price per sq ft by number of amenities")
axes[0].set_xlabel("Number of amenities")
axes[0].set_ylim(0, by_count.max() * 1.2)
sns.barplot(x=list(differences.keys()), y=list(differences.values()), color="seagreen", ax=axes[1])
axes[1].set_title("Price per sq ft difference: with vs without the amenity (Rs)")
plt.suptitle("Q19: Amenities and Price per Sq Ft")
save_chart("eda_q19_amenities.png")

# Q20: public transport accessibility vs price per sq ft and investment potential
order = ["Low", "Medium", "High"]
transport = df.groupby("Public_Transport_Accessibility")[PPSF].mean().reindex(order)
good_share = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(order) * 100
# The Good_Investment rule gives +1 point for High transport, so remove that point to see the real effect
score_without_transport = df["Investment_Score"] - (df["Public_Transport_Accessibility"] == "High").astype(int)
good_without = (score_without_transport >= 3).groupby(df["Public_Transport_Accessibility"]).mean().reindex(order) * 100
print("Q20 Average price per sq ft (Rs) by transport access:", transport.round(0).to_dict())
print("   Good_Investment share (%) by transport access:", good_share.round(1).to_dict())
print("   Same share if the transport point is removed from the rule:", good_without.round(1).to_dict())
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=transport.index, y=transport.values, hue=transport.index, legend=False, ax=axes[0])
axes[0].set_title("Average price per sq ft (Rs)")
sns.barplot(x=good_share.index, y=good_share.values, hue=good_share.index, legend=False, ax=axes[1])
axes[1].set_title("Good Investment share (%)")
plt.suptitle("Q20: Public Transport Accessibility")
save_chart("eda_q20_transport.png")