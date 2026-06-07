import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Prédiction des Résultats Médicaux-SAID EL YAZAMI", layout="wide")

# Load models and preprocessing objects
@st.cache_resource
def load_resources():
    model = joblib.load('models/medical_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_resources()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()





# Centered Header with Logo and Title
logo_path = os.path.join(os.path.dirname(__file__), "EHTP LOGO.png")
if os.path.exists(logo_path):
    import base64
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    logo_base64 = get_base64_of_bin_file(logo_path)
    header_html = f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="120">
        <h2 style="margin-bottom: 0;">Master Data Engineering</h2>
        <p style="font-weight: bold; margin-bottom: 0;">Projet final du Module 5 « Machine Learning - 2025/2026 »</p>
        <p>Projet réalisé par Mr.SAID EL YAZAMI</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; white-space: nowrap;'></h1>", unsafe_allow_html=True)







# Sidebar for navigation
st.sidebar.title("MENU")
page = st.sidebar.radio("", ["Home", "Prediction", "Data Analysis"])

# Load data for Analysis page
@st.cache_data
def load_data():
    return pd.read_csv('healthcare_dataset.csv')

df = load_data()

if page == "Home":
    st.title("Analyse et Prédiction des Résultats Médicaux")
    st.markdown("""
    Cette application prédit le résultat des examens médicaux pour les patients hospitalisés en se basant sur leurs données cliniques.
    
    ### Aperçu du Jeu de Données
    Le jeu de données comprend des informations telles que :
    - Données démographiques du patient (Âge, Genre).
    - État de santé.
    - Type d'admission.
    - Traitements médicamenteux prescrits.
    - Informations de couverture d'assurance.
    
    ### Objectif
    Prédire si le résultat d'un examen sera Normal, Anormal ou Non concluant.
    """)


elif page == "Prediction":
    st.title("Prédiction pour un Nouveau Patient")
    st.write("Saisissez les informations du patient pour prédire le résultat de l'examen médical.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 0, 100, 30)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        blood_type = st.selectbox("Blood Type", encoders['Blood Type'].classes_)
        medical_condition = st.selectbox("Medical Condition", encoders['Medical Condition'].classes_)
        
    with col2:
        insurance = st.selectbox("Insurance Provider", encoders['Insurance Provider'].classes_)
        admission_type = st.selectbox("Admission Type", encoders['Admission Type'].classes_)
        medication = st.selectbox("Medication", encoders['Medication'].classes_)
        billing_amount = st.number_input("Billing Amount ($)", min_value=0.0, value=10000.0)
        days_hospitalized = st.number_input("Expected Days of Hospitalization", min_value=0, value=7)

    if st.button("Prédire le résultat"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [encoders['Gender'].transform([gender])[0]],
            'Blood Type': [encoders['Blood Type'].transform([blood_type])[0]],
            'Medical Condition': [encoders['Medical Condition'].transform([medical_condition])[0]],
            'Insurance Provider': [encoders['Insurance Provider'].transform([insurance])[0]],
            'Billing Amount': [billing_amount],
            'Admission Type': [encoders['Admission Type'].transform([admission_type])[0]],
            'Medication': [encoders['Medication'].transform([medication])[0]],
            'Days_Hospitalized': [days_hospitalized]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction_idx = model.predict(input_scaled)[0]
        prediction_label = encoders['Test Results'].classes_[prediction_idx]
        
        # Display result
        st.subheader(f"Prediction: **{prediction_label}**")
        
        # Probability
        probs = model.predict_proba(input_scaled)[0]
        prob_df = pd.DataFrame({
            'Result': encoders['Test Results'].classes_,
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index('Result'))

elif page == "Data Analysis":
    st.title("Analyse Exploratoire des Données")
    
    st.write("### Distribution de la Variable Cible")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Test Results', hue='Test Results', palette='viridis', legend=False, ax=ax1)
    st.pyplot(fig1)
    
    st.write("### État de santé vs Résultats des examens")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='Medical Condition', hue='Test Results', palette='magma', ax=ax2)
    st.pyplot(fig2)
    
    st.write("### Distribution des montants facturés selon les résultats des examens")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Test Results', y='Billing Amount', hue='Test Results', palette='coolwarm', legend=False, ax=ax3)
    st.pyplot(fig3)
    
    st.write("### Distribution par âge")
    fig4, ax4 = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax4)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Projet réalisé par Mr.SAID EL YAZAMI - MSDE 2025/2026</p>", unsafe_allow_html=True)
st.markdown("---")

