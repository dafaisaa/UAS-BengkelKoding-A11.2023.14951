import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('top_features.pkl')
    return model, scaler, features

model, scaler, top_features = load_model()

st.title("📊 Customer Churn Prediction App")
st.write("Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan (Churn) menggunakan model Machine Learning.")

st.sidebar.header("Informasi Model")
st.sidebar.info("Model: Random Forest Classifier\nPreprocessing: StandardScaler + SMOTE\nFitur: 15 Fitur Teratas")

st.subheader("Masukkan Data Pelanggan")
with st.form("input_form"):
    user_inputs = {}
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(top_features):
        target_col = col1 if idx % 2 == 0 else col2
        
        # PERBAIKAN LOGIKA: Hanya fitur biner hasil One-Hot Encoding yang jadi dropdown Ya/Tidak
        if "type_" in feature or "channel_" in feature or "gender_" in feature:
            with target_col:
                user_inputs[feature] = st.selectbox(
                    label=feature, 
                    options=[0, 1], 
                    format_func=lambda x: "Ya / Positif (1)" if x == 1 else "Tidak / Negatif (0)"
                )
        else:
            # Sisa fiturnya (seperti age, total_spent, dll) akan menjadi input angka
            with target_col:
                user_inputs[feature] = st.number_input(label=feature, value=0.0, step=0.1)
                
    submit_button = st.form_submit_button("Prediksi Status Churn")

if submit_button:
    input_df = pd.DataFrame([user_inputs], columns=top_features)
    
    # Proses scaling sekarang akan berhasil karena fiturnya sama-sama 15
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("Hasil Analisis")
    
    if prediction == 1:
        st.error(f"⚠️ **PELANGGAN BERPOTENSI CHURN**")
        st.write(f"Probabilitas Churn: **{prediction_proba[1] * 100:.2f}%**")
        st.write("Rekomendasi: Berikan penawaran khusus atau diskon retensi untuk mempertahankan pelanggan ini.")
    else:
        st.success(f"✅ **PELANGGAN CENDERUNG BERTAHAN (RETAIN)**")
        st.write(f"Probabilitas Bertahan: **{prediction_proba[0] * 100:.2f}%**")
        st.write("Rekomendasi: Lanjutkan pelayanan standar dan tawarkan program loyalitas.")