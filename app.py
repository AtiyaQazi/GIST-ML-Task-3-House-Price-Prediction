import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

MODEL_PATH = "models/house_price_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Please run model_training.py first.")
    st.stop()

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🏠 House Price Prediction System")

st.write(
    "Enter the house details below to estimate the house price."
)

st.divider()

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

st.subheader("Enter House Information")

income = st.number_input(
    "Average Area Income",
    min_value=0.0,
    value=80000.0,
    step=1000.0
)

house_age = st.number_input(
    "Average Area House Age",
    min_value=0.0,
    value=7.0,
    step=0.1
)

rooms = st.number_input(
    "Average Area Number of Rooms",
    min_value=1.0,
    value=8.0,
    step=1.0
)

bedrooms = st.number_input(
    "Average Area Number of Bedrooms",
    min_value=1.0,
    value=4.0,
    step=1.0
)

population = st.number_input(
    "Area Population",
    min_value=0.0,
    value=45000.0,
    step=1000.0
)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

rooms_per_bedroom = rooms / bedrooms if bedrooms != 0 else 0
population_per_room = population / rooms if rooms != 0 else 0

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if st.button("Predict House Price", use_container_width=True):

    input_data = pd.DataFrame({
        "Avg. Area Income": [income],
        "Avg. Area House Age": [house_age],
        "Avg. Area Number of Rooms": [rooms],
        "Avg. Area Number of Bedrooms": [bedrooms],
        "Area Population": [population],
        "Rooms_per_Bedroom": [rooms_per_bedroom],
        "Population_per_Room": [population_per_room]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction generated successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.2f}"
    )

    st.write("### Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Average Area Income",
            "House Age",
            "Rooms",
            "Bedrooms",
            "Population",
            "Rooms per Bedroom",
            "Population per Room"
        ],
        "Value": [
            income,
            house_age,
            rooms,
            bedrooms,
            population,
            rooms_per_bedroom,
            population_per_room
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("Model Information")

    st.write("**Final Model:** Linear Regression")

    st.write("**R² Score:** 0.9179")

    st.write("**MAE:** $80,914.27")

    st.write("**RMSE:** $100,478.05")

    st.divider()

    st.write("### Project Pipeline")

    st.write("1. Data Cleaning")
    st.write("2. Exploratory Data Analysis")
    st.write("3. Feature Engineering")
    st.write("4. Model Training")
    st.write("5. Hyperparameter Tuning")
    st.write("6. Model Evaluation")
    st.write("7. Prediction")