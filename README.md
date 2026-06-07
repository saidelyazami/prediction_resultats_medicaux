# Projet N° 4 : Prédiction du résultat d'analyse médicale d'un patient hospitalisé

Ce projet vise à prédire le résultat d'un test médical (Normal, Abnormal, Inconclusive) pour des patients hospitalisés en utilisant des techniques de Machine Learning.

## Structure du Projet
- `healthcare_dataset.csv` : Le jeu de données utilisé.
- `train_model.py` : Script Python pour le nettoyage des données, l'entraînement du modèle (Random Forest) et la sauvegarde des artefacts.
- `app.py` : Application Streamlit pour l'interface utilisateur.
- `models/` : Dossier contenant le modèle entraîné, le scaler et les encodeurs.
- `requirements.txt` : Liste des dépendances.

## Installation
Assurez-vous d'avoir Python installé, puis installez les dépendances :
```bash
pip install pandas scikit-learn streamlit joblib matplotlib seaborn
```

## Utilisation
1. Entraîner le modèle (optionnel, déjà fait) :
```bash
python train_model.py
```
2. Lancer l'application Streamlit :
```bash
streamlit run app.py
```

## Modèle
Le modèle utilisé est un **Random Forest Classifier**. Bien que les données synthétiques présentent une variance élevée, le pipeline de traitement (encodage, calcul de la durée de séjour, normalisation) est robuste et applicable à des données réelles.
