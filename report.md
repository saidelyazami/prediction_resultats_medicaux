# Rapport de Projet ML - MSDE 7
## Projet 4 : Prédiction du résultat d'analyse médicale d'un patient hospitalisé

### 1. Compréhension de la problématique
L'objectif est d'aider les professionnels de santé en prédisant automatiquement le résultat d'un bilan médical (Normal, Abnormal, ou Inconclusive) à partir des données démographiques et médicales du patient.
- **Tâche ML** : Classification multi-classe.
- **Données** : 10 000 dossiers hospitaliers synthétiques.

### 2. Analyse exploratoire des données (EDA)
Nous avons analysé :
- La distribution des classes (équilibrée autour de 33% chacune).
- La distribution de l'âge (uniforme entre 18 et 85 ans).
- L'impact des conditions médicales et des types d'admission sur les résultats.
- Note : S'agissant de données synthétiques générées aléatoirement, les corrélations sont quasi-nulles.

### 3. Pré-processing des données
- **Feature Engineering** : Calcul de la durée du séjour (`Stay_Duration`) à partir des dates d'admission et de sortie.
- **Sélection de features** : Suppression des colonnes non informatives (`Name`, `Doctor`, `Hospital`).
- **Encodage** : OneHotEncoding pour les variables catégorielles et LabelEncoding pour la cible.
- **Normalisation** : StandardScaler pour les variables numériques (`Age`, `Billing Amount`, `Stay_Duration`).

### 4. Construction des modèles ML
Nous avons testé 10 algorithmes :
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. AdaBoost
6. Gradient Boosting
7. XGBoost
8. LightGBM
9. CatBoost
10. SVM

L'exactitude (accuracy) se situe autour de 33-34% pour tous les modèles, ce qui est attendu pour des données synthétiques sans relations statistiques.

### 5. Tuning et Choix du modèle final
Le modèle **Random Forest** a été sélectionné pour sa robustesse et optimisé par `GridSearchCV`. Le pipeline complet incluant le pré-traitement a été sérialisé dans `model.joblib`.

### 6. Déploiement
Une application interactive a été développée avec **Streamlit** (`app.py`), permettant de saisir les données d'un patient et d'obtenir une prédiction avec son niveau de confiance.
