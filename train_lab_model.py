import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("gastrointestinal_disease_dataset.csv")

# Target column
TARGET_COLUMN = "Disease_Class"

# Split
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# Convert categorical → numeric
X = pd.get_dummies(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# Save
joblib.dump((model, le, X.columns), "lab_model.pkl")

print("Model trained successfully")
