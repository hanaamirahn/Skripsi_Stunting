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

        # 1. Encoding gender
        gender_encoded = gender_encoder.transform([gender])[0]

        # 2. Input numerik (URUTAN IDENTIK TRAINING)
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
    
        # 3. Scaling
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
    
        # ===============================
        # 5. PREDIKSI (HARUS DULU)
        # ===============================
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]
    
        # ===============================
        # 6. TAMPILKAN PROBABILITAS
        # ===============================
        st.markdown("### 📈 Probabilitas Kelas")
        st.write(f"🟢 Tidak Stunting (0): **{proba[0]:.2f}**")
        st.write(f"🔴 Stunting (1): **{proba[1]:.2f}**")
    
        st.progress(int(proba[1] * 100))
    
        # ===============================
        # 7. INTERPRETASI FINAL
        # ===============================
        if prediction == 1:
            st.error("⚠️ **BERISIKO STUNTING**")
        else:
            st.success("✅ **TIDAK BERISIKO STUNTING**")
    
        st.markdown("---")

# =====================================================
# 🧠 TAB 2 — MODEL & INFORMASI
# =====================================================
import os
from PIL import Image

with tab2:

    st.subheader("🧠 Informasi Model")

    st.markdown("""
    ### Cara Kerja Model Klasifikasi

    Proses klasifikasi risiko stunting pada balita dimulai dari tahap
    **pra-pemrosesan data**, meliputi pembersihan data, encoding variabel
    kategorik (jenis kelamin), serta normalisasi fitur numerik menggunakan
    *scaler*.

    Dataset yang mengalami ketidakseimbangan kelas kemudian ditangani
    menggunakan metode **Synthetic Minority Over-sampling Technique (SMOTE)**,
    sehingga distribusi data antar kelas menjadi lebih seimbang.

    Model utama yang digunakan adalah **Random Forest**. Untuk meningkatkan
    performa model, khususnya pada nilai F1-score, dilakukan optimasi
    hyperparameter menggunakan **Particle Swarm Optimization (PSO)**.
    Model hasil optimasi ini digunakan sebagai model terbaik dalam sistem.
    """)

    st.markdown("---")

    # =====================================================
    # VISUAL EVALUATION (SIDE BY SIDE)
    # =====================================================
    col_cm, col_cr = st.columns(2)

    # ==========================
    # CONFUSION MATRIX
    # ==========================
    with col_cm:
        st.markdown("### Confusion Matrix")

        cm_pso_path = "assets/cm_pso.png"

        if os.path.exists(cm_pso_path):
            cm_img = Image.open(cm_pso_path)
            st.image(
                cm_img,
                caption="Confusion Matrix Model RF + SMOTE + PSO",
                use_container_width=True
            )
        else:
            st.warning("📂 Gambar confusion matrix belum tersedia.")

        st.markdown("""
        Pada **kelas 0 (tidak stunting)**, terdapat **220 data** yang berhasil
        diprediksi dengan benar sebagai kelas 0, sedangkan **189 data**
        masih keliru diprediksi sebagai kelas 1. Hal ini menunjukkan bahwa
        kemampuan model dalam membedakan kelas 0 masih perlu ditingkatkan.

        Sementara itu, pada **kelas 1 (stunting)**, model menunjukkan kinerja
        yang lebih baik dengan **1474 data** berhasil diprediksi dengan benar
        dan hanya **117 data** yang salah diklasifikasikan. Hal ini menandakan
        bahwa model cukup konsisten dalam mengenali pola pada kelas stunting.
        """)

    # ==========================
    # GARIS PEMBATAS TENGAH
    # ==========================
    with col_div:
        st.markdown(
            """
            <div style="
                height: 100%;
                border-left: 2px solid #e0e0e0;
                margin: auto;
            "></div>
            """,
            unsafe_allow_html=True
        )

    # ==========================
    # CLASSIFICATION REPORT
    # ==========================
    with col_cr:
        st.markdown("### Classification Report")

        cr_path = "assets/cr_img.png"

        if os.path.exists(cr_path):
            cr_img = Image.open(cr_path)
            st.image(
                cr_img,
                caption="Classification Report Model RF + SMOTE + PSO",
                use_container_width=True
            )
        else:
            st.warning("📂 Gambar classification report belum tersedia.")

        st.markdown("""
        *Classification report* menunjukkan bahwa model mencapai
        **akurasi sebesar 0.85 (85%)** dalam mengklasifikasikan dua kelas.

        Pada **kelas 0**, model memiliki **precision 0.65**, **recall 0.54**,
        dan **F1-score 0.59**, yang menunjukkan bahwa performa model pada kelas
        ini masih belum optimal.

        Sebaliknya, pada **kelas 1**, model menunjukkan performa yang lebih
        kuat dengan **precision 0.89**, **recall 0.93**, dan **F1-score 0.91**,
        yang mencerminkan konsistensi model dalam mengenali data stunting.
        """)

    st.markdown("""
    <div style="text-align:center; margin-top:40px; color:gray;">
        <b>Hana Amirah Natasya</b><br>
        Program Studi Teknik Informatika<br>
        Fakultas Matematika dan Ilmu Pengetahuan Alam<br>
        Universitas Negeri Semarang<br>
        2026
    </div>
    """, unsafe_allow_html=True)
