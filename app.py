
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fraud_xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    data = pd.read_csv("data/creditcard.csv")
    return model, scaler, data

model, scaler, data = load_artifacts()
explainer = shap.Explainer(model, scaler.transform(data.drop('Class', axis=1)))

st.set_page_config(page_title="XAI Fraud Detection", layout="wide")
st.title("💳 Explainable AI: Fraud Detection in Finance")

st.sidebar.header("Transaction Input")
option = st.sidebar.selectbox("Choose input method:", ["Use random fraud sample", "Manual entry"])

if option == "Use random fraud sample":
    sample = data[data['Class'] == 1].sample(1, random_state=42).drop("Class", axis=1)
else:
    columns = data.drop("Class", axis=1).columns
    sample = {}
    for col in columns:
        sample[col] = st.sidebar.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    sample = pd.DataFrame([sample])

scaled = scaler.transform(sample)
pred_class = model.predict(scaled)[0]
pred_prob = model.predict_proba(scaled)[0, 1]

st.subheader("PREDICTION")
st.write(f"**Predicted Class:** {'Fraudulent' if pred_class == 1 else 'Legitimate'}")
st.write(f"**Fraud Probability:** {pred_prob:.4f}")

st.subheader("SHAP EXPLANATION")
shap_values = explainer(scaled)
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig)

