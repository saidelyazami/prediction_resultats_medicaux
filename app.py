import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and label encoder
@st.cache_resource
def load_resources():
    model = joblib.load('model.joblib')
    le = joblib.load('label_encoder.joblib')
    return model, le

model, le = load_resources()

st.set_page_config(page_title="Medical Analysis Predictor", layout="centered")

st.title("🏥 Medical Analysis Result Predictor")
st.markdown("""
Predict the result of a medical analysis (Normal, Abnormal, or Inconclusive) 
based on patient demographics and medical conditions.
""")

st.sidebar.header("Patient Data Input")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 40)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    blood_type = st.sidebar.selectbox('Blood Type', ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'))
    condition = st.sidebar.selectbox('Medical Condition', ('Diabetes', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis', 'Cancer'))
    insurance = st.sidebar.selectbox('Insurance Provider', ('Aetna', 'Blue Cross', 'Cigna', 'UnitedHealthcare', 'Medicare'))
    billing = st.sidebar.number_input('Billing Amount', min_value=0.0, value=25000.0)
    room = st.sidebar.number_input('Room Number', min_value=1, value=101)
    admission_type = st.sidebar.selectbox('Admission Type', ('Emergency', 'Elective', 'Urgent'))
    medication = st.sidebar.selectbox('Medication', ('Aspirin', 'Ibuprofen', 'Penicillin', 'Paracetamol', 'Lipitor'))
    stay_duration = st.sidebar.slider('Stay Duration (days)', 1, 30, 7)

    data = {
        'Age': age,
        'Gender': gender,
        'Blood Type': blood_type,
        'Medical Condition': condition,
        'Insurance Provider': insurance,
        'Billing Amount': billing,
        'Room Number': room,
        'Admission Type': admission_type,
        'Medication': medication,
        'Stay_Duration': stay_duration
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Patient Summary")
st.write(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    res_label = le.inverse_transform(prediction)[0]
    confidence = np.max(prediction_proba) * 100
    
    st.subheader("Prediction Result")
    if res_label == 'Normal':
        st.success(f"Result: {res_label}")
    elif res_label == 'Abnormal':
        st.error(f"Result: {res_label}")
    else:
        st.warning(f"Result: {res_label}")
        
    st.write(f"**Confidence Level:** {confidence:.2f}%")
    
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    st.bar_chart(prob_df.T)

st.info("Note: This model is trained on synthetic data for demonstration purposes.")
