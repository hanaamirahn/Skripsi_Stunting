import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Klasifikasi Risiko Stunting",
    page_icon="🧒",
    layout="wide"
)

# -------------------------------
# LOAD MODEL & TOOLS
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/rf_pso.pkl")
    scaler = joblib.load("model/scaler.pkl")
    gender_encoder = joblib.load("model/gender_encoder.pkl")
    return model, scaler, gender_encoder

model, scaler, gender_encoder = load_model()

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Klasifikasi", "🧠 Model & Evaluasi"]
)

# ======================================================
# 🏠 HOME
# ======================================================
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
    Stunting merupakan kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis,
    terutama pada 1.000 hari pertama kehidupan. Dampak stunting tidak hanya memengaruhi
    pertumbuhan fisik, tetapi juga perkembangan kognitif, produktivitas di masa depan,
    serta meningkatkan risiko penyakit degeneratif.

    Oleh karena itu, diperlukan sistem klasifikasi yang akurat untuk membantu
    deteksi dini risiko stunting sehingga intervensi dapat dilakukan lebih cepat dan tepat sasaran.
    """)

    st.info("👉 Gunakan menu **Klasifikasi** untuk melakukan prediksi risiko stunting.")

# ======================================================
# 📊 KLASIFIKASI
# ======================================================
elif menu == "📊 Klasifikasi":

    st.header("📊 Klasifikasi Risiko Stunting")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia (bulan)", min_value=0, max_value=60)
        birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.5, max_value=6.0, step=0.1)

    with col2:
        birth_length = st.number_input("Panjang Lahir (cm)", min_value=30.0, max_value=60.0, step=0.1)
        body_weight = st.number_input("Berat Badan Saat Ini (kg)", min_value=2.0, max_value=25.0, step=0.1)
        body_length = st.number_input("Panjang Badan Saat Ini (cm)", min_value=40.0, max_value=120.0, step=0.1)

    if st.button("🔍 Klasifikasi"):

    gender_encoded = gender_encoder.transform([gender])[0]

    input_df = pd.DataFrame([{
        "Age": age,
        "Birth Weight": birth_weight,
        "Birth Length": birth_length,
        "Body Weight": body_weight,
        "Body Length": body_length
    }])

    input_scaled = scaler.transform(input_df)

    final_input = pd.DataFrame(
        np.column_stack([gender_encoded, input_scaled]),
        columns=['Gender', 'Age', 'Birth Weight', 'Birth Length', 'Body Weight', 'Body Length']
    )

    prediction = model.predict(final_input)[0]


# ======================================================
# 🧠 MODEL & EVALUASI
# ======================================================
elif menu == "🧠 Model & Evaluasi":

    st.header("🧠 Evaluasi Model")

    st.subheader("📌 Skenario 3: RF + SMOTE + PSO")
    st.write("""
    Model terbaik diperoleh dari kombinasi Random Forest,
    SMOTE untuk menangani ketidakseimbangan data,
    serta Particle Swarm Optimization (PSO) untuk optimasi hyperparameter.
    """)

    cm_pso = Image.open("assets/cm_pso.png")
    st.image(cm_pso, caption="Confusion Matrix — RF + SMOTE + PSO", use_container_width=True)

    st.subheader("📊 Perbandingan Model")
    st.write("""
    - **RF Original**: performa dasar tanpa penanganan imbalance
    - **RF + SMOTE**: recall meningkat signifikan
    - **RF + SMOTE + PSO**: F1-score terbaik dan model paling stabil
    """)

    cm_compare = Image.open("assets/cm_compare.png")
    st.image(cm_compare, caption="Perbandingan Confusion Matrix", use_container_width=True)

    st.success("✅ Model RF + SMOTE + PSO dipilih sebagai model terbaik.")
