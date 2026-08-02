# ============================================
# Breast Cancer Prediction System
# app.py
# ============================================

import streamlit as st
import pandas as pd
import joblib

# ============================================
# Load Model and Scaler
# ============================================

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    color:#0B5394;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:20px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================

st.sidebar.title("📋 Project Information")

st.sidebar.markdown("""
### 👨‍💻 Developer
Sanajit

### 🤖 Algorithm
Random Forest

### 💻 Language
Python

### 📚 Framework
Streamlit

### 🧠 Machine Learning
Scikit-learn

### 📊 Dataset
Breast Cancer Wisconsin Dataset
""")

# ============================================
# Main Title
# ============================================

st.markdown(
    '<p class="main-title">🩺 Breast Cancer Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Machine Learning Based Tumor Classification</p>',
    unsafe_allow_html=True
)

st.info(
    """
This application predicts whether a breast tumor is:

✅ **Benign (Non-Cancerous)**

or

⚠️ **Malignant (Cancerous)**

using a trained Machine Learning model.
"""
)

st.markdown("---")

# ============================================
# Input Section
# ============================================

st.header("📝 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        value=17.99,
        help="Average radius of the cell nucleus"
    )

    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        value=10.38,
        help="Texture of the cell"
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        value=122.80,
        help="Average perimeter of the cell"
    )

with col2:

    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0,
        value=1001.00,
        help="Average area of the cell"
    )

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        value=0.1184,
        format="%.4f",
        help="Average smoothness of the cell"
    )

st.markdown("")

# ============================================
# Prediction
# ============================================

if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness
        ]],
        columns=[
            "mean_radius",
            "mean_texture",
            "mean_perimeter",
            "mean_area",
            "mean_smoothness"
        ]
    )

    with st.spinner("Analyzing patient data..."):

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

    st.markdown("---")

    st.header("📋 Prediction Result")

    if prediction[0] == 1:

        st.success("✅ Prediction: Benign (Non-Cancerous)")

        st.balloons()

    else:

        st.error("⚠️ Prediction: Malignant (Cancerous)")

    # Prediction Confidence
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_scaled)

        confidence = probability.max() * 100

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

    st.markdown("---")

    st.subheader("📄 Patient Details")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# ============================================
# About Project
# ============================================

st.markdown("---")

with st.expander("📖 About This Project"):

    st.markdown("""
### 🩺 Breast Cancer Prediction System

This project predicts whether a breast tumor is **Benign** or **Malignant**
using Machine Learning.

### 🔍 Algorithms Tested

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

### 🎯 Objective

To help classify breast tumors based on medical measurements using a trained Machine Learning model.

**Note:** This application is intended for educational purposes and should not replace professional medical diagnosis.
""")

# ============================================
# Footer
# ============================================

st.markdown("---")

st.markdown(
    '<p class="footer">Developed using ❤️ Python, Scikit-learn and Streamlit</p>',
    unsafe_allow_html=True
)