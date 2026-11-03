"""Real Estate Investment Advisor - run with: streamlit run app.py"""
import sys
from pathlib import Path

import altair as alt
import joblib
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


data = get_data()
clf, reg = get_models()
options = {c: sorted(data[c].dropna().unique()) for c in CATEGORICAL_FEATURES}

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a good investment and its estimated price in 5 years.")

tab_predict, tab_market, tab_model = st.tabs(["Predict", "Market explorer", "Model insights"])

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
            price = st.number_input("Price (lakhs)", float(data.Price_in_Lakhs.min()), float(data.Price_in_Lakhs.max()), 100.0)
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
        row = pd.DataFrame([{
            "State": state, "City": city, "Property_Type": property_type, "Furnished_Status": furnished,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
            "Price_per_SqFt_Rupees": price * 100000 / size, "Age_of_Property": age,
            "Floor_No": floor_no, "Total_Floors": total_floors,
            "Floor_Inconsistent": int(floor_no > total_floors),
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Amenity_Count": len(chosen),
            **{f"Amenity_{a}": int(a in chosen) for a in AMENITIES},
        }])
        row["Infrastructure_Score"] = compute_infrastructure_score(row)
        row = row[FEATURES]

        probability = float(clf.predict_proba(row)[0, 1])
        future_price = float(reg.predict(row)[0])
        growth = (future_price / price - 1) * 100

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

        city_avg = data.loc[data["City"] == city, "Price_per_SqFt_Rupees"].mean()
        this_psf = price * 100000 / size
        st.caption(f"Your price per sq ft: ₹{this_psf:,.0f} | {city} average: ₹{city_avg:,.0f} "
                   f"({(this_psf / city_avg - 1) * 100:+.0f}% vs city average)")
        if floor_no > total_floors:
            st.info("Floor number is higher than total floors, so this input was flagged as inconsistent.")
        st.caption("Models are trained on a dataset whose targets are derived from price, size and location, "
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
        "Regressor": ["Ridge", "Random forest (best)", "Hist. gradient boosting"],
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