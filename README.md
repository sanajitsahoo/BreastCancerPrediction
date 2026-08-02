# 🩺 Breast Cancer Prediction System

A Machine Learning project that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using patient medical data.

---

## 📌 Project Overview

This project uses Machine Learning algorithms to classify breast cancer based on medical measurements. It also includes a user-friendly web application built with Streamlit.

---

## 🚀 Features

- Predicts whether a tumor is Benign or Malignant
- Interactive Streamlit web application
- Data preprocessing and feature scaling
- Machine Learning model training
- Model saving using Joblib
- Easy-to-use interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```text
BreastCancerPrediction/
│
├── dataset/
│   └── Breast_cancer_data.csv
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebook/
│   └── analysis.ipynb
│
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sanajitsahoo/BreastCancerPrediction.git
```

### 2. Open the Project

```bash
cd BreastCancerPrediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Dataset

**Breast Cancer Wisconsin Dataset**

Features used:

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness

Target:

- Benign
- Malignant

---

## 🤖 Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## 📸 Application

The application allows users to:

- Enter patient details
- Predict the diagnosis
- View prediction confidence
- Display entered patient information

---

## 👨‍💻 Developer

**Sanajit Sahoo**

B.Tech Computer Science Engineering

---

## 📄 License

This project is created for educational purposes.
