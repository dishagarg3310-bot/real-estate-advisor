"""Real Estate Investment Advisor - run with: streamlit run app.py"""
import re
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    compute_infrastructure_score,
    load_data,
)

AMENITIES = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
HORIZON_YEARS = 5

EDA_QUESTIONS = {
    "Q1. Distribution of property prices": "eda_q01_price_distribution.png",
    "Q2. Distribution of property sizes": "eda_q02_size_distribution.png",
    "Q3. Price per sq ft by property type": "eda_q03_ppsf_by_type.png",
    "Q4. Property size vs price": "eda_q04_size_vs_price.png",
    "Q5. Outliers in price per sq ft and size": "eda_q05_outliers.png",
    "Q6. Average price per sq ft by state": "eda_q06_ppsf_by_state.png",
    "Q7. Average property price by city": "eda_q07_price_by_city.png",
    "Q8. Median age of properties by locality": "eda_q08_age_by_locality.png",
    "Q9. BHK distribution across cities": "eda_q09_bhk_by_city.png",
    "Q10. Price trends, top 5 expensive localities": "eda_q10_top5_locality_trend.png",
    "Q11. Correlation between numeric features": "eda_q11_correlation_heatmap.png",
    "Q12. Nearby schools vs price per sq ft": "eda_q12_schools_vs_ppsf.png",
    "Q13. Nearby hospitals vs price per sq ft": "eda_q13_hospitals_vs_ppsf.png",
    "Q14. Price by furnished status": "eda_q14_price_by_furnished.png",
    "Q15. Price per sq ft by facing direction": "eda_q15_ppsf_by_facing.png",
    "Q16. Properties by owner type": "eda_q16_owner_type.png",
    "Q17. Properties by availability status": "eda_q17_availability.png",
    "Q18. Does parking space affect price?": "eda_q18_parking.png",
    "Q19. Amenities vs price per sq ft": "eda_q19_amenities.png",
    "Q20. Public transport vs price per sq ft": "eda_q20_transport.png",
}

st.set_page_config(page_title="Real Estate Advisor", page_icon="🏠", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_models():
    return joblib.load("models/best_classification.joblib"), joblib.load("models/best_regression.joblib")


@st.cache_data
def get_importance(task: str):
    path = Path("reports") / f"feature_importance_{task}.csv"
    return pd.read_csv(path) if path.exists() else None


def importance_chart(df: pd.DataFrame, label: str):
    return alt.Chart(df.head(12)).mark_bar().encode(
        x=alt.X("Importance:Q", title=label),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=["Feature", "Importance"],
    )


def get_finding(q_number: str):
    """Return question + answer text for one EDA question from reports/eda_findings.md."""
    path = Path("reports") / "eda_findings.md"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        clean = line.replace("*", "").lstrip("#-• ").strip()
        if re.match(rf"^{q_number}\.\s", clean):
            parts = [clean]
            for nxt in lines[i + 1:]:
                nxt_clean = nxt.replace("*", "").strip()
                if not nxt_clean:
                    if len(parts) > 1:
                        break
                    continue
                if re.match(r"^\d+\.\s", nxt_clean) or nxt_clean.startswith(("Section", "Overall")):
                    break
                parts.append(nxt_clean)
            text = " ".join(parts).split("Chart:")[0].strip()
            question, _, answer = text.partition("?")
            return f"**{question.strip()}?**\n\n{answer.strip()}" if answer.strip() else text
    return None

def build_frame(base: dict, prices) -> pd.DataFrame:
    """Model-ready rows for one property at one or more prices (lakhs)."""
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    df = pd.DataFrame([base] * len(prices))
    df["Price_in_Lakhs"] = prices
    df["Price_per_SqFt_Rupees"] = prices * 100000 / df["Size_in_SqFt"]
    df["Infrastructure_Score"] = compute_infrastructure_score(df)
    return df[FEATURES]


data = get_data()
clf, reg = get_models()
options = {c: sorted(data[c].dropna().unique()) for c in CATEGORICAL_FEATURES}

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a good investment and its estimated price in 5 years.")

tab_predict, tab_market, tab_eda, tab_batch, tab_model = st.tabs(
    ["Predict", "Market explorer", "EDA", "Batch", "Model insights"]
)
# ---------------------------------------------------------------- Predict
with tab_predict:
    with st.form("property"):
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox("State", options["State"])
            cities = sorted(data.loc[data["State"] == state, "City"].unique())
            city = st.selectbox("City", cities)
            property_type = st.selectbox("Property type", options["Property_Type"])
            furnished = st.selectbox("Furnished status", options["Furnished_Status"])
            owner = st.selectbox("Owner type", options["Owner_Type"])
            availability = st.selectbox("Availability", options["Availability_Status"])
        with c2:
            bhk = st.number_input("BHK", int(data.BHK.min()), int(data.BHK.max()), 2)
            size = st.number_input("Size (sq ft)", float(data.Size_in_SqFt.min()), float(data.Size_in_SqFt.max()), 1200.0)
            price_in = st.number_input("Price (lakhs)", float(data.Price_in_Lakhs.min()), float(data.Price_in_Lakhs.max()), 100.0)
            age = st.number_input("Age of property (years)", 0, int(data.Age_of_Property.max()), 5)
            total_floors = st.number_input("Total floors", 1, int(data.Total_Floors.max()), 10)
            floor_no = st.number_input("Floor number", 0, int(data.Floor_No.max()), 3)
        with c3:
            transport = st.selectbox("Public transport access", options["Public_Transport_Accessibility"])
            parking = st.selectbox("Parking space", options["Parking_Space"])
            security = st.selectbox("Security", options["Security"])
            facing = st.selectbox("Facing", options["Facing"])
            schools = st.slider("Nearby schools", 1, 10, 5)
            hospitals = st.slider("Nearby hospitals", 1, 10, 5)
            chosen = st.multiselect("Amenities", AMENITIES)
        submitted = st.form_submit_button("Analyse property", type="primary")

    if submitted:
        st.session_state["analysis"] = {
            "price": float(price_in),
            "base": {
                "State": state, "City": city, "Property_Type": property_type, "Furnished_Status": furnished,
                "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
                "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
                "BHK": bhk, "Size_in_SqFt": float(size), "Age_of_Property": age,
                "Floor_No": floor_no, "Total_Floors": total_floors,
                "Floor_Inconsistent": int(floor_no > total_floors),
                "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
                "Amenity_Count": len(chosen),
                **{f"Amenity_{a}": int(a in chosen) for a in AMENITIES},
            },
        }

    analysis = st.session_state.get("analysis")
    if analysis:
        base, price = analysis["base"], analysis["price"]
        city = base["City"]
        row = build_frame(base, price)

        probability = float(clf.predict_proba(row)[0, 1])
        future_price = float(reg.predict(row)[0])
        gain = future_price - price
        growth = (future_price / price - 1) * 100
        cagr = ((future_price / price) ** (1 / HORIZON_YEARS) - 1) * 100

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Good investment probability", f"{probability * 100:.1f}%")
        m2.metric("Estimated price in 5 years", f"{future_price:,.1f} lakhs")
        m3.metric("Expected 5-year growth", f"{growth:+.1f}%")

        confidence = max(probability, 1 - probability)
        st.progress(confidence, text=f"Model confidence in this verdict: {confidence * 100:.1f}%")
        if probability >= 0.5:
            st.success("Verdict: Good investment")
        else:
            st.warning("Verdict: Not a good investment")

        # ---- Investment metrics
        city_rows = data[data["City"] == city]
        city_median_price = float(city_rows["Price_in_Lakhs"].median())
        city_median_growth = float((city_rows["Future_Price_5Y"] / city_rows["Price_in_Lakhs"] - 1).median() * 100)
        city_avg_psf = float(city_rows["Price_per_SqFt_Rupees"].mean())
        this_psf = price * 100000 / base["Size_in_SqFt"]

        st.subheader("Investment metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("5-year CAGR", f"{cagr:.2f}% / year")
        k2.metric("Total gain in 5 years", f"{gain:+,.1f} lakhs")
        k3.metric(f"Price vs {city} median", f"{(price / city_median_price - 1) * 100:+.0f}%",
                  help=f"{city} median price: {city_median_price:,.1f} lakhs")
        k4.metric(f"Growth vs {city} median", f"{growth - city_median_growth:+.1f} pts",
                  help=f"{city} median 5-year growth: {city_median_growth:.1f}%")
        st.caption(f"Your price per sq ft: ₹{this_psf:,.0f} | {city} average: ₹{city_avg_psf:,.0f} "
                   f"({(this_psf / city_avg_psf - 1) * 100:+.0f}% vs city average)")
        if base["Floor_Inconsistent"]:
            st.info("Floor number is higher than total floors, so this input was flagged as inconsistent.")

        # ---- What-if chart
        st.subheader("What if the price changes?")
        pmin, pmax = float(data.Price_in_Lakhs.min()), float(data.Price_in_Lakhs.max())
        grid = np.linspace(max(pmin, price * 0.5), min(pmax, price * 1.5), 30)
        grid = np.unique(np.append(grid, price))
        whatif_rows = build_frame(base, grid)
        whatif = pd.DataFrame({
            "Price": grid,
            "Probability": clf.predict_proba(whatif_rows)[:, 1] * 100,
            "Future_price": reg.predict(whatif_rows),
        })
        marker = alt.Chart(pd.DataFrame({"Price": [price]})).mark_rule(strokeDash=[5, 4], color="#ff4b4b").encode(x="Price:Q")
        w1, w2 = st.columns(2)
        w1.altair_chart(
            alt.Chart(whatif).mark_line(point=True).encode(
                x=alt.X("Price:Q", title="Price (lakhs)", scale=alt.Scale(zero=False)),
                y=alt.Y("Probability:Q", title="Good investment probability (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Price:Q", format=",.1f"), alt.Tooltip("Probability:Q", format=".1f")],
            ) + marker,
            use_container_width=True,
        )
        w2.altair_chart(
            alt.Chart(whatif).mark_line(point=True, color="#2ca02c").encode(
                x=alt.X("Price:Q", title="Price (lakhs)", scale=alt.Scale(zero=False)),
                y=alt.Y("Future_price:Q", title="Estimated price in 5 years (lakhs)", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("Price:Q", format=",.1f"), alt.Tooltip("Future_price:Q", format=",.1f")],
            ) + marker,
            use_container_width=True,
        )
        st.caption("Red dashed line = your price. Only the price changes; every other input stays the same. "
                   "Models are trained on a dataset whose targets are derived from price, size and location, "
                   "so treat results as a demo, not financial advice.")

# ---------------------------------------------------------------- Market explorer
with tab_market:
    st.subheader("Filter properties")
    pmin, pmax = float(data.Price_in_Lakhs.min()), float(data.Price_in_Lakhs.max())
    smin, smax = int(data.Size_in_SqFt.min()), int(data.Size_in_SqFt.max())
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", options["State"])
    bhks = f2.multiselect("BHK", sorted(data["BHK"].unique()))
    price_range = f3.slider("Price (lakhs)", pmin, pmax, (pmin, pmax), step=5.0)
    size_range = f4.slider("Size (sq ft)", smin, smax, (smin, smax), step=50)

    view = data[data["Price_in_Lakhs"].between(*price_range) & data["Size_in_SqFt"].between(*size_range)]
    if states:
        view = view[view["State"].isin(states)]
    if bhks:
        view = view[view["BHK"].isin(bhks)]

    if view.empty:
        st.warning("No properties match these filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Properties", f"{len(view):,}")
        k2.metric("Average price", f"{view.Price_in_Lakhs.mean():,.1f} lakhs")
        k3.metric("Average price per sq ft", f"₹{view.Price_per_SqFt_Rupees.mean():,.0f}")
        k4.metric("Good investments", f"{view.Good_Investment.mean() * 100:.1f}%")

        st.subheader("Heatmap: average price per sq ft by city and BHK")
        heat = view.groupby(["City", "BHK"], as_index=False)["Price_per_SqFt_Rupees"].mean()
        st.altair_chart(
            alt.Chart(heat).mark_rect().encode(
                x=alt.X("BHK:O"),
                y=alt.Y("City:N", sort="-color", title=None),
                color=alt.Color("Price_per_SqFt_Rupees:Q", title="Avg ₹/sq ft", scale=alt.Scale(scheme="viridis")),
                tooltip=["City", "BHK", alt.Tooltip("Price_per_SqFt_Rupees:Q", format=",.0f")],
            )
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Price per sq ft by property age")
            trend = view.groupby("Age_of_Property", as_index=False)["Price_per_SqFt_Rupees"].mean()
            st.altair_chart(
                alt.Chart(trend).mark_line(point=True).encode(
                    x=alt.X("Age_of_Property:Q", title="Age (years)"),
                    y=alt.Y("Price_per_SqFt_Rupees:Q", title="Avg ₹/sq ft", scale=alt.Scale(zero=False)),
                    tooltip=["Age_of_Property", alt.Tooltip("Price_per_SqFt_Rupees:Q", format=",.0f")],
                )
            )
        with right:
            st.subheader("Today vs in 5 years (top 10 cities by price)")
            top = (view.groupby("City")[["Price_in_Lakhs", "Future_Price_5Y"]].mean()
                   .nlargest(10, "Price_in_Lakhs").reset_index())
            long = top.melt("City", var_name="When", value_name="Price_lakhs")
            long["When"] = long["When"].map({"Price_in_Lakhs": "Today", "Future_Price_5Y": "In 5 years"})
            st.altair_chart(
                alt.Chart(long).mark_bar().encode(
                    x=alt.X("City:N", sort=None, title=None),
                    xOffset="When:N",
                    y=alt.Y("Price_lakhs:Q", title="Avg price (lakhs)"),
                    color=alt.Color("When:N", title=None),
                    tooltip=["City", "When", alt.Tooltip("Price_lakhs:Q", format=",.1f")],
                )
            )

        st.subheader("Matching properties")
        st.dataframe(
            view.head(200)[["State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                            "Price_per_SqFt_Rupees", "Age_of_Property", "Good_Investment", "Future_Price_5Y"]],
            hide_index=True,
        )
        st.caption(f"Showing the first {min(200, len(view))} of {len(view):,} matching properties.")

# ---------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Exploratory data analysis")
    question = st.selectbox("Choose a question", list(EDA_QUESTIONS))
    chart_path = Path("reports") / EDA_QUESTIONS[question]
    finding = get_finding(question.split(".")[0].replace("Q", ""))

    col_chart, col_text = st.columns([3, 2])
    with col_chart:
        if chart_path.exists():
            st.image(str(chart_path), width=620)
        else:
            st.warning(f"Chart not found: {chart_path}")
    with col_text:
        if finding:
            st.markdown("**Finding**")
            st.markdown(finding)
        else:
            st.info("No finding text for this question.")

    findings_path = Path("reports") / "eda_findings.md"
    if findings_path.exists():
        with st.expander("Read all EDA findings"):
            st.markdown(findings_path.read_text(encoding="utf-8"))
# ---------------------------------------------------------------- Batch
with tab_batch:
    st.subheader("Batch prediction")
    st.write("Upload a CSV with one property per row. It must contain these columns: "
             + ", ".join(f"`{c}`" for c in FEATURES))
    st.caption("Tip: `Infrastructure_Score`, `Price_per_SqFt_Rupees`, `Amenity_Count` and `Floor_Inconsistent` "
               "are recalculated automatically if missing.")

    sample_path = Path("docs") / "sample_batch.csv"
    if sample_path.exists():
        st.download_button("Download sample CSV", sample_path.read_bytes(),
                           file_name="sample_batch.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)

        # Recalculate derived columns when absent
        if "Price_per_SqFt_Rupees" not in batch and {"Price_in_Lakhs", "Size_in_SqFt"} <= set(batch):
            batch["Price_per_SqFt_Rupees"] = batch["Price_in_Lakhs"] * 100000 / batch["Size_in_SqFt"]
        if "Floor_Inconsistent" not in batch and {"Floor_No", "Total_Floors"} <= set(batch):
            batch["Floor_Inconsistent"] = (batch["Floor_No"] > batch["Total_Floors"]).astype(int)
        amenity_flags = [f"Amenity_{a}" for a in AMENITIES]
        if "Amenity_Count" not in batch and set(amenity_flags) <= set(batch):
            batch["Amenity_Count"] = batch[amenity_flags].sum(axis=1)
        if "Infrastructure_Score" not in batch and {
            "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility"
        } <= set(batch):
            batch["Infrastructure_Score"] = compute_infrastructure_score(batch)

        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif len(batch) > 20000:
            st.error("Please upload at most 20,000 rows.")
        else:
            X = batch[FEATURES]
            result = batch.copy()
            result["Good_Investment_Probability_%"] = (clf.predict_proba(X)[:, 1] * 100).round(1)
            result["Verdict"] = np.where(result["Good_Investment_Probability_%"] >= 50,
                                         "Good investment", "Not a good investment")
            result["Future_Price_5Y_Lakhs"] = reg.predict(X).round(2)
            result["Growth_%"] = ((result["Future_Price_5Y_Lakhs"] / result["Price_in_Lakhs"] - 1) * 100).round(1)

            s1, s2, s3 = st.columns(3)
            s1.metric("Properties", f"{len(result):,}")
            s2.metric("Good investments", f"{(result['Verdict'] == 'Good investment').mean() * 100:.1f}%")
            s3.metric("Average 5-year growth", f"{result['Growth_%'].mean():+.1f}%")

            front = ["Verdict", "Good_Investment_Probability_%", "Future_Price_5Y_Lakhs", "Growth_%"]
            st.dataframe(result[front + [c for c in result.columns if c not in front]], hide_index=True)
            st.download_button("Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv", mime="text/csv", type="primary")
# ---------------------------------------------------------------- Model insights
with tab_model:
    st.subheader("Model comparison (test set, 50,000 rows)")
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame({
        "Classifier": ["Logistic regression", "Random forest", "Hist. gradient boosting (best)"],
        "Accuracy": [0.8610, 0.9961, 0.9989],
        "F1": [0.8556, 0.9960, 0.9989],
        "ROC AUC": [0.9495, 0.9995, 1.0000],
    }), hide_index=True)
    b.dataframe(pd.DataFrame({
        "Regressor": ["Ridge", "Random forest", "Hist. gradient boosting (deployed)"],
        "RMSE (lakhs)": [9.51, 0.36, 1.21],
        "MAE (lakhs)": [6.72, 0.22, 0.97],
        "R²": [0.9978, 1.0000, 1.0000],
    }), hide_index=True)

    cm_path = Path("reports") / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion matrix (classifier)")
        st.image(str(cm_path), width=420)

    st.subheader("Feature importance")
    i1, i2 = st.columns(2)
    clf_imp, reg_imp = get_importance("classification"), get_importance("regression")
    if clf_imp is not None:
        i1.caption("Good investment classifier (drop in ROC AUC when the feature is shuffled)")
        i1.altair_chart(importance_chart(clf_imp, "Importance"))
    if reg_imp is not None:
        i2.caption("5-year price regressor (drop in R² when the feature is shuffled)")
        i2.altair_chart(importance_chart(reg_imp, "Importance"))
    st.caption("Targets are derived from price, size and location, so those features dominate by design.")