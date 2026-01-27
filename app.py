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
        # 2. DATAFRAME NUMERIK (URUTAN SAMA)
        # ===============================
        input_df = pd.DataFrame([{
            "Age": age,
            "Birth Weight": birth_weight,
            "Birth Length": birth_length,
            "Body Weight": body_weight,
            "Body Length": body_length
        }])

        # ===============================
        # 3. SCALING
        # ===============================
        input_scaled = scaler.transform(input_df)

        # ===============================
        # 4. FINAL INPUT (IDENTIK TRAINING)
        # ===============================
        final_input = pd.DataFrame(
            np.column_stack([gender_encoded, input_scaled]),
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
        # 5. CEK CLASS ORDER (ANTI SALAH)
        # ===============================
        classes = model.classes_

        if list(classes) == [0, 1]:
            stunting_index = 1
        else:
            stunting_index = 0

        # ===============================
        # 6. PROBABILITAS
        # ===============================
        proba = model.predict_proba(final_input)[0][stunting_index]

        st.write(f"Probabilitas Stunting: **{proba:.2f}**")
        st.progress(int(proba * 100))

        # ===============================
        # 7. KEPUTUSAN (2 KELAS)
        # ===============================
        THRESHOLD = 0.70  # lebih realistis utk dataset imbalance

        if proba >= THRESHOLD:
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
