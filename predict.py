# ===================================
# Breast Cancer Prediction
# predict.py
# ===================================

import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

print("=" * 40)
print("BREAST CANCER PREDICTION SYSTEM")
print("=" * 40)

# User Input
mean_radius = float(input("Enter Mean Radius: "))
mean_texture = float(input("Enter Mean Texture: "))
mean_perimeter = float(input("Enter Mean Perimeter: "))
mean_area = float(input("Enter Mean Area: "))
mean_smoothness = float(input("Enter Mean Smoothness: "))

# Create DataFrame
data = pd.DataFrame(
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

# Scale input
scaled_data = scaler.transform(data)

# Prediction
prediction = model.predict(scaled_data)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Benign (Non-Cancerous)")
else:
    print("Malignant (Cancerous)")