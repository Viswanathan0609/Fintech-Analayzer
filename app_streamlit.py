import streamlit as st
import pandas as pd
import joblib
import os
from preprocessing import load_and_preprocess
from fraud_model import train_classical_model
from quantum_model import train_quantum_model

st.title("🔮 Quantum FinTech Analyzer")

uploaded_file = st.file_uploader("Upload Financial Dataset", type=["csv"])

if uploaded_file:
    st.success("Dataset uploaded successfully!")

    X_train, X_test, y_train, y_test = load_and_preprocess(uploaded_file)

    if st.button("Train Classical Model"):
        acc = train_classical_model(X_train, X_test, y_train, y_test)
        st.write(f"Classical Model Accuracy: {acc*100:.2f}%")

    if st.button("Train Quantum Model"):
        acc = train_quantum_model(X_train, X_test, y_train, y_test)
        st.write(f"Quantum Model Accuracy: {acc*100:.2f}%")

    st.subheader("Fraud Risk Prediction")

    model_path = "models/classical_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)

        sample = X_test[0:1]
        prediction = model.predict(sample)

        if prediction[0] == 1:
            st.error("⚠️ High Fraud Risk Transaction")
        else:
            st.success("✅ Low Risk Transaction")
    else:
        st.info("Please train the Classical Model first to enable prediction.")
