import streamlit as st
import joblib
import numpy as np
from scipy import sparse
import tensorflow as tf
import os
import pandas as pd

# ======================================================
# App Configuration
# ======================================================
st.set_page_config(page_title="DeepCSAT — E-commerce Satisfaction Predictor", layout="centered")
st.title("🧠 DeepCSAT — Predict Customer Satisfaction Score")

# ======================================================
# Load preprocessing artifacts
# ======================================================
tfidf = joblib.load('artifacts/tfidf.joblib')
scaler = joblib.load('artifacts/scaler.joblib')
ohe = joblib.load('artifacts/ohe.joblib')
schema = joblib.load('artifacts/schema.joblib')

text_col = schema['text_col']
numeric_cols = schema['numeric_cols']
categorical_cols = schema['categorical_cols']

# ======================================================
# Load dataset to extract category options
# ======================================================
cat_values = {}
try:
    df = pd.read_csv('eCommerce_Customer_support_data.csv')
    for col in categorical_cols:
        # Take unique non-null values (up to 50 options for UI performance)
        unique_vals = sorted(df[col].dropna().unique().tolist())[:50]
        if len(unique_vals) == 0:
            unique_vals = ["NA"]
        cat_values[col] = unique_vals
    st.success("📂 Loaded category values from dataset.")
except Exception as e:
    cat_values = {col: ["NA"] for col in categorical_cols}
    st.warning(f"⚠️ Could not load category values from data.csv — defaulted to 'NA'. ({e})")

# ======================================================
# Safe Keras 3 model loading
# ======================================================
model_path_keras = "artifacts/deepcsat_model.keras"
model_path_h5 = "artifacts/deepcsat_model.h5"
model_path_tf = "artifacts/deepcsat_model"   # folder (legacy SavedModel)

model = None

if os.path.exists(model_path_keras):
    model = tf.keras.models.load_model(model_path_keras)
    st.info("✅ Loaded .keras model format (Keras 3 recommended).")
elif os.path.exists(model_path_h5):
    model = tf.keras.models.load_model(model_path_h5)
    st.info("✅ Loaded legacy .h5 model format.")
elif os.path.isdir(model_path_tf):
    from keras.layers import TFSMLayer
    model = TFSMLayer(model_path_tf, call_endpoint='serving_default')
    st.warning("⚠️ Loaded TensorFlow SavedModel via TFSMLayer (inference-only).")
else:
    st.error("❌ No trained model found. Please train and save the model first.")
    st.stop()

# ======================================================
# UI Section
# ======================================================
st.markdown("### ✍️ Customer Input")

# --- Text input ---
review = st.text_area("Customer Remarks", placeholder="Type customer feedback here...")

# --- Numeric inputs ---
st.markdown("### 🔢 Numeric Features")
numeric_values = []
for col in numeric_cols:
    val = st.number_input(f"{col}", value=0.0, key=f"num_{col}")
    numeric_values.append(val)

# --- Categorical dropdowns ---
st.markdown("### 🧩 Categorical Features")
categorical_values = []
for col in categorical_cols:
    options = cat_values.get(col, ["NA"])
    choice = st.selectbox(f"{col}", options=options, key=f"cat_{col}")
    categorical_values.append(choice)

# ======================================================
# Prediction
# ======================================================
if st.button("🔮 Predict CSAT Score"):
    if not review.strip():
        st.warning("⚠️ Please enter a customer remark before predicting.")
        st.stop()

    # Transform inputs
    X_text = tfidf.transform([review])
    X_num = scaler.transform([numeric_values])
    X_cat = ohe.transform([categorical_values])

    # Combine all feature parts
    X = sparse.hstack([X_text, X_num, X_cat]).tocsr()
    row = X.getrow(0).toarray().astype('float32')

    # Predict (supports both Model and TFSMLayer)
    try:
        pred = model.predict(row, verbose=0)[0, 0]
    except AttributeError:
        pred = model(row).numpy()[0, 0]

    st.success(f"🎯 Predicted CSAT Score: **{pred:.2f}**")

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption("DeepCSAT © 2025 — Customer Satisfaction Prediction for E-commerce")
