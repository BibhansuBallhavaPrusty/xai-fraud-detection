# tests/test_pipeline.py - Unit tests for XAI Fraud Detection Pipeline

import pytest
import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = "models/fraud_xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/creditcard.csv"

@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    data = pd.read_csv(DATA_PATH)
    return model, scaler, data

# Test 1: Check if model and scaler exist
def test_artifacts_exist():
    assert os.path.exists(MODEL_PATH), "Model file missing"
    assert os.path.exists(SCALER_PATH), "Scaler file missing"

# Test 2: Validate model predictions on a sample
def test_model_prediction(artifacts):
    model, scaler, data = artifacts
    sample = data.drop('Class', axis=1).sample(1, random_state=42)
    scaled_sample = scaler.transform(sample)
    pred = model.predict(scaled_sample)
    assert pred[0] in [0, 1], "Prediction must be binary"

# Test 3: Ensure scaler transforms shape correctly
def test_scaler_shape(artifacts):
    _, scaler, data = artifacts
    original_shape = data.drop('Class', axis=1).shape
    scaled = scaler.transform(data.drop('Class', axis=1))
    assert scaled.shape == original_shape, "Scaler output shape mismatch"

# Test 4: SHAP value shape matches input
def test_shap_shape(artifacts):
    import shap
    model, scaler, data = artifacts
    X = scaler.transform(data.drop('Class', axis=1))
    explainer = shap.Explainer(model, X)
    shap_vals = explainer(X[:1])
    assert shap_vals.shape[1] == X.shape[1], "SHAP value dimension mismatch"