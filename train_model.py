import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Chargement des données
df = pd.read_csv('healthcare_dataset.csv')

# Nettoyage et Prétraitement des données
# 1. La suppression des colonnes inutiles
cols_to_drop = ['Name', 'Doctor', 'Hospital', 'Room Number']
df = df.drop(columns=cols_to_drop)

# 2. La conversation des dates et en va calculer la durée du séjour
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Days_Hospitalized'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# La suppression des dates d'origine
df = df.drop(columns=['Date of Admission', 'Discharge Date'])

# 3. L'encodage des variables catégorielles
cat_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 4. L'encodage cible
le_target = LabelEncoder()
df['Test Results'] = le_target.fit_transform(df['Test Results'])
encoders['Test Results'] = le_target

# 5. La séparation des fonctionnalités (features) et la cible (target)
X = df.drop(columns=['Test Results'])
y = df['Test Results']

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. L'entraînement de mon modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 9. L'évaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 10. La sauvegarde de mon modèle et les objets de prétraitement
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/medical_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')

print("Modèle et encodeurs sauvegardés dans le dossier models/.")
