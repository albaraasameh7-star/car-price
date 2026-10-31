import streamlit as st
import joblib
import pandas as pd

# ---------- Page setup ----------
st.set_page_config(page_title="My ML Model", page_icon="🤖", layout="centered")
st.title("🤖 My Model Predictor")
st.write("Enter the values below and click **Predict**.")

# ---------- Load model (cached so it loads once) ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------- Inputs (CHANGE these to match your model's features) ----------
col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
with col2:
    feature3 = st.slider("Feature 3", 0, 100, 50)
    feature4 = st.selectbox("Feature 4", ["A", "B", "C"])

# Encode categorical input the same way you did in training
mapping = {"A": 0, "B": 1, "C": 2}

# ---------- Predict ----------
if st.button("Predict"):
    try:
        X = pd.DataFrame(
            [[feature1, feature2, feature3, mapping[feature4]]],
            columns=["feature1", "feature2", "feature3", "feature4"],  # same names/order as training
        )
        prediction = model.predict(X)[0]
        st.success(f"Prediction: **{prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.bar_chart(pd.DataFrame(proba, columns=["Probability"]))
    except Exception as e:
        st.error(f"Something went wrong: {e}")
