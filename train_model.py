# ================================
# STEP 1 : Import Libraries
# ================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# ================================
# STEP 2 : Load Dataset
# ================================

df = pd.read_csv("dataset/Breast_cancer_data.csv")

print("Dataset Loaded Successfully\n")

# ================================
# STEP 3 : Features and Target
# ================================

X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]

# ================================
# STEP 4 : Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Data Split Completed\n")

# ================================
# STEP 5 : Feature Scaling
# ================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Feature Scaling Completed\n")

# ================================
# STEP 6 : Machine Learning Models
# ================================

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC()

}

# ================================
# STEP 7 : Train and Evaluate
# ================================

best_accuracy = 0
best_model = None
best_model_name = ""

print("Model Accuracy\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name} : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

        # ================================
# STEP 8 : Best Model
# ================================

print("\nBest Model :", best_model_name)

print("Accuracy :", round(best_accuracy * 100, 2), "%")

# ================================
# STEP 9 : Save Model
# ================================

joblib.dump(best_model, "models/model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully")

print("Scaler Saved Successfully")