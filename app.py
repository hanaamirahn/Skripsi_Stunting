import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Klasifikasi Risiko Stunting",
    page_icon="🧒",
    layout="wide"
)

# =====================================================
# LOAD MODEL (NAMA FILE TIDAK DIUBAH)
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("model/rf_pso.pkl")
    scaler = joblib.load("model/scaler.pkl")
    gender_encoder = joblib.load("model/gender_encoder.pkl")
    return model, scaler, gender_encoder

model, scaler, gender_encoder = load_model()

# =====================================================
# HEADER UTAMA
# =====================================================
st.markdown(
    """
    <h2 style="text-align:center;">
    Klasifikasi Risiko Stunting pada Balita
    </h2>
    <p style="text-align:center; color:gray;">
    Klasifikasi Ini Menggunakan Kombinasi Model Random Forest, SMOTE, dan Particle Swarm Optimization
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# NAVIGASI ATAS
# =====================================================
tab1, tab2 = st.tabs(["🔍 Klasifikasi", "📖 Model & Informasi"])

# =====================================================
# 📊 TAB 1 — KLASIFIKASI
# =====================================================
with tab1:

    st.subheader("Input Data")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia (bulan)", 0, 60, 24)
        birth_weight = st.number_input("Berat Lahir (kg)", 0.5, 6.0, 3.0, step=0.1)

    with col2:
        birth_length = st.number_input("Panjang Lahir (cm)", 30.0, 60.0, 49.0, step=0.1)
        body_weight = st.number_input("Berat Badan Saat Ini (kg)", 2.0, 25.0, 10.0, step=0.1)
        body_length = st.number_input("Panjang Badan Saat Ini (cm)", 40.0, 120.0, 80.0, step=0.1)

    st.markdown("---")

    if st.button("🔍 Lakukan Klasifikasi", use_container_width=True):

        # 1. Encoding gender (SAMA seperti training)
        gender_encoded = gender_encoder.transform([gender])[0]

        # 2. Input numerik (URUTAN HARUS IDENTIK)
        numeric_input = pd.DataFrame([[
            age,
            birth_weight,
            birth_length,
            body_weight,
            body_length
        ]], columns=[
            "Age",
            "Birth Weight",
            "Birth Length",
            "Body Weight",
            "Body Length"
        ])

        # 3. Scaling (PAKAI scaler training)
        numeric_scaled = scaler.transform(numeric_input)

        # 4. Final input
        final_input = pd.DataFrame(
            np.hstack([[gender_encoded], numeric_scaled[0]]).reshape(1, -1),
            columns=[
                "Gender",
                "Age",
                "Birth Weight",
                "Birth Length",
                "Body Weight",
                "Body Length"
            ]
        )

        # 5. Interpretasi final (0 = tidak, 1 = stunting)
        if prediction == 1:
            st.error("⚠️ **BERISIKO STUNTING**")
        else:
            st.success("✅ **TIDAK BERISIKO STUNTING**")

        # 6. Prediksi
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]

        st.markdown("### 📈 Probabilitas Kelas")
        st.write(f"🟢 Tidak Stunting (0): **{proba[0]:.2f}**")
        st.write(f"🔴 Stunting (1): **{proba[1]:.2f}**")

        st.progress(int(proba[1] * 100))

        st.markdown("---")

# =====================================================
# 🧠 TAB 2 — MODEL & INFORMASI
# =====================================================
with tab2:

    st.subheader("🧠 Informasi Model")

    st.write("""
    Aplikasi ini menggunakan algoritma **Random Forest** yang dioptimasi
    menggunakan **Particle Swarm Optimization (PSO)** serta penyeimbangan data
    dengan **SMOTE** untuk meningkatkan sensitivitas terhadap kasus stunting.
    """)

    st.markdown("### 🎯 Model Terbaik")
    st.info("Random Forest + SMOTE + PSO")

    st.markdown("### 📊 Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        cm_pso = Image.open("assets/cm_pso.png")
        st.image(cm_pso, caption="Confusion Matrix RF + SMOTE + PSO", use_container_width=True)

    with col2:
        cm_compare = Image.open("assets/cm_compare.png")
        st.image(cm_compare, caption="Perbandingan Kinerja Model", use_container_width=True)

    st.markdown("---")
    st.caption("© 2026 — Hana Amirah Natasya | Universitas Negeri Semarang")
