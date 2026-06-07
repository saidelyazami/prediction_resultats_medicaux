import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
df = pd.read_csv('healthcare_dataset.csv')

# Data Cleaning / Preprocessing
# 1. Drop irrelevant columns
cols_to_drop = ['Name', 'Doctor', 'Hospital', 'Room Number']
df = df.drop(columns=cols_to_drop)

# 2. Convert dates and calculate Length of Stay
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Days_Hospitalized'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Drop original dates
df = df.drop(columns=['Date of Admission', 'Discharge Date'])

# 3. Encoding Categorical Variables
cat_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 4. Target Encoding
le_target = LabelEncoder()
df['Test Results'] = le_target.fit_transform(df['Test Results'])
encoders['Test Results'] = le_target

# 5. Split Features and Target
X = df.drop(columns=['Test Results'])
y = df['Test Results']

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scaling (Optional for Random Forest, but good for others)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Evaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 10. Save Model and Preprocessing objects
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/medical_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')

print("Model and encoders saved in models/ folder.")
