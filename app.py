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
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("model/rf_pso.pkl")
    scaler = joblib.load("model/scaler.pkl")
    gender_encoder = joblib.load("model/gender_encoder.pkl")
    return model, scaler, gender_encoder

model, scaler, gender_encoder = load_model()

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Klasifikasi", "🧠 Model & Evaluasi"]
)

# =====================================================
# 🏠 HOME
# =====================================================
if menu == "🏠 Home":

    st.markdown(
        """
        <h2 style='text-align:center'>
        OPTIMASI HYPERPARAMETER RANDOM FOREST<br>
        MENGGUNAKAN PARTICLE SWARM OPTIMIZATION DAN SMOTE<br>
        UNTUK KLASIFIKASI RISIKO STUNTING PADA BALITA
        </h2>
        <p style='text-align:center'>
        <b>Hana Amirah Natasya</b><br>
        Program Studi Sarjana Teknik Informatika<br>
        Fakultas Matematika dan Ilmu Pengetahuan Alam<br>
        Universitas Negeri Semarang
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📌 Mengapa Stunting Harus Dideteksi Sejak Dini?")
    st.write("""
    Stunting merupakan kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis.
    Deteksi dini penting untuk mencegah dampak jangka panjang pada kesehatan, kognitif,
    dan kualitas hidup anak.
    """)

# =====================================================
# 📊 KLASIFIKASI
# =====================================================
elif menu == "📊 Klasifikasi":

    st.header("📊 Klasifikasi Risiko Stunting")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia (bulan)", 0, 60, 24)
        birth_weight = st.number_input("Berat Lahir (kg)", 0.5, 6.0, 3.0, step=0.1)

    with col2:
        birth_length = st.number_input("Panjang Lahir (cm)", 30.0, 60.0, 49.0, step=0.1)
        body_weight = st.number_input("Berat Badan Saat Ini (kg)", 2.0, 25.0, 10.0, step=0.1)
        body_length = st.number_input("Panjang Badan Saat Ini (cm)", 40.0, 120.0, 80.0, step=0.1)

    if st.button("🔍 Klasifikasi"):

        # ===============================
        # 1. ENCODING GENDER
        # ===============================
        gender_encoded = gender_encoder.transform([gender])[0]

        # ===============================
        # 2. DATA NUMERIK (URUTAN HARUS SAMA DENGAN TRAINING)
        # ===============================
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

        # ===============================
        # 3. SCALING (PAKAI SCALER TRAINING)
        # ===============================
        numeric_scaled = scaler.transform(numeric_input)

        # ===============================
        # 4. FINAL INPUT (IDENTIK SAAT TRAINING)
        # ===============================
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

        # ===============================
        # 5. PREDIKSI
        # ===============================
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]

        st.write(f"Probabilitas Tidak Stunting (0): **{proba[0]:.2f}**")
        st.write(f"Probabilitas Stunting (1): **{proba[1]:.2f}**")

        # ===============================
        # 6. INTERPRETASI LABEL (FINAL)
        # ===============================
        if prediction == 1:
            st.error("⚠️ BERISIKO STUNTING")
        else:
            st.success("✅ TIDAK BERISIKO STUNTING")

# =====================================================
# 🧠 MODEL & EVALUASI
# =====================================================
elif menu == "🧠 Model & Evaluasi":

    st.header("🧠 Evaluasi Model")

    st.subheader("Skenario Terbaik: RF + SMOTE + PSO")
    st.write("""
    Model ini dipilih karena menghasilkan F1-score terbaik
    serta mampu meningkatkan sensitivitas terhadap kasus stunting.
    """)

    cm_pso = Image.open("assets/cm_pso.png")
    st.image(cm_pso, caption="Confusion Matrix RF + SMOTE + PSO", use_container_width=True)

    cm_compare = Image.open("assets/cm_compare.png")
    st.image(cm_compare, caption="Perbandingan Model", use_container_width=True)
