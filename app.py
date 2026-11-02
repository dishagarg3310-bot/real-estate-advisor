"""Real Estate Investment Advisor - run with: streamlit run app.py"""
import sys
from pathlib import Path

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


data = get_data()
clf, reg = get_models()
options = {c: sorted(data[c].dropna().unique()) for c in CATEGORICAL_FEATURES}
city_to_state = data.drop_duplicates("City").set_index("City")["State"].to_dict()

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a good investment and its estimated price in 5 years.")

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

    if probability >= 0.5:
        st.success("Verdict: Good investment")
    else:
        st.warning("Verdict: Not a good investment")
    if floor_no > total_floors:
        st.info("Floor number is higher than total floors, so this input was flagged as inconsistent.")
    st.caption("Models are trained on a dataset whose targets are derived from price, size and location, "
               "so treat results as a demo, not financial advice.")