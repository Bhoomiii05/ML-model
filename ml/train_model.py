import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("loan_data.csv")

# 🔹 Drop unnecessary columns (like Loan_ID)
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

# 🔹 Handle '3+' in Dependents column (Convert to Numeric)
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

# 🔹 Encode categorical values
encoder = LabelEncoder()
categorical_columns = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))  # Ensure data is string before encoding

# 🔹 Features (X) and Target (Y)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model training completed successfully!")
