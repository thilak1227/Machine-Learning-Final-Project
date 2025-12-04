import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# =======================
# CONFIGURATION
# =======================
MODEL_FILENAME = 'xgboost_kidney_model.pkl'
SCALER_FILENAME = 'kidney_scaler.pkl'
FEATURE_NAMES = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
CATEGORICAL_MAPS = {
    'rbc': {'normal': 1, 'abnormal': 0, 'missing': 2},
    'pc': {'normal': 1, 'abnormal': 0, 'missing': 2},
    'pcc': {'present': 1, 'notpresent': 0, 'missing': 2},
    'ba': {'present': 1, 'notpresent': 0, 'missing': 2},
    'htn': {'yes': 2, 'no': 0, 'missing': 1},
    'dm': {'yes': 2, 'no': 0, 'missing': 1},
    'cad': {'yes': 2, 'no': 0, 'missing': 1},
    'appet': {'good': 1, 'poor': 0, 'missing': 2},
    'pe': {'yes': 2, 'no': 0, 'missing': 1},
    'ane': {'yes': 2, 'no': 0, 'missing': 1}
}


# =======================
# LOAD MODEL AND SCALER
# =======================

@st.cache_resource
def load_assets():
    try:
        # Load the saved model
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        
        # Load the saved scaler
        with open(SCALER_FILENAME, 'rb') as file:
            scaler = pickle.load(file)
            
        return model, scaler
    except FileNotFoundError:
        st.error(f"Required files ({MODEL_FILENAME} or {SCALER_FILENAME}) not found.")
        st.info("Please run the `train_kidney_model.py` script first.")
        st.stop()

model, scaler = load_assets()

# =======================
# APP UI
# =======================

st.set_page_config(page_title="Kidney Disease Prediction", layout="wide")

st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("---")
st.subheader("Patient Input Features")
st.write("Use the controls below to enter patient clinical measurements.")

# =======================
# INPUT WIDGETS
# =======================

# Helper function to map user-friendly labels to numerical values
def get_categorical_value(feature, user_input):
    # This logic matches the LabelEncoder values derived from your original data
    if user_input == 'Yes': return 2
    if user_input == 'No': return 0
    if user_input == 'Present': return 1
    if user_input == 'Not Present': return 0
    if user_input == 'Normal': return 1
    if user_input == 'Abnormal': return 0
    if user_input == 'Good': return 1
    if user_input == 'Poor': return 0
    return user_input # For numerical features or if no match is found

# Layout inputs using Streamlit columns for a better look (like your friend's app)
col1, col2, col3 = st.columns(3)

# Column 1: Core Biometrics
with col1:
    age = st.number_input("Age (years)", 1, 100, 45)
    bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
    al = st.slider("Albumin", 0, 5, 1)

# Column 2: Blood Tests
with col2:
    su = st.slider("Sugar", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random (mg/dl)", 50, 500, 120)
    bu = st.number_input("Blood Urea (mg/dl)", 10, 390, 40)
    sc = st.number_input("Serum Creatinine (mg/dl)", 0.4, 50.0, 1.2, format="%.1f")

# Column 3: Red Blood Cell / Status
with col3:
    sod = st.number_input("Sodium (mEq/L)", 110, 150, 137)
    pot = st.number_input("Potassium (mEq/L)", 2.5, 8.0, 4.0, format="%.1f")
    hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, 14.0, format="%.1f")
    rbc = get_categorical_value('rbc', st.selectbox("Red Blood Cells", ['Normal', 'Abnormal']))

# Use a container for the less critical binary inputs
with st.expander("More Health Indicators"):
    col4, col5, col6 = st.columns(3)
    
    with col4:
        htn = get_categorical_value('htn', st.selectbox("Hypertension (HTN)", ['No', 'Yes']))
        dm = get_categorical_value('dm', st.selectbox("Diabetes Mellitus (DM)", ['No', 'Yes']))
        cad = get_categorical_value('cad', st.selectbox("Coronary Artery Disease (CAD)", ['No', 'Yes']))
    
    with col5:
        appet = get_categorical_value('appet', st.selectbox("Appetite", ['Good', 'Poor']))
        pe = get_categorical_value('pe', st.selectbox("Pedal Edema (PE)", ['No', 'Yes']))
        ane = get_categorical_value('ane', st.selectbox("Anemia (ANE)", ['No', 'Yes']))

    with col6:
        pc = get_categorical_value('pc', st.selectbox("Pus Cell", ['Normal', 'Abnormal']))
        pcc = get_categorical_value('pcc', st.selectbox("Pus Cell Clumps", ['Not Present', 'Present']))
        ba = get_categorical_value('ba', st.selectbox("Bacteria", ['Not Present', 'Present']))
        
        # Placeholder for pcv, wc, rc, which are manually entered below:
        pcv = st.number_input("Packed Cell Volume (PCV)", 10, 60, 40)
        wc = st.number_input("White Blood Cell Count (wc)", 2200, 26400, 8000)
        rc = st.number_input("Red Blood Cell Count (rc)", 2.1, 8.0, 5.0, format="%.1f")


# --- 4. PREDICTION LOGIC ---

if st.button("PREDICT KIDNEY DISEASE STATUS"):
    # Create the input DataFrame, ensuring columns match the training features
    input_data = pd.DataFrame({
        'age': [age], 'bp': [bp], 'sg': [sg], 'al': [al], 'su': [su],
        'rbc': [rbc], 'pc': [pc], 'pcc': [pcc], 'ba': [ba], 'bgr': [bgr],
        'bu': [bu], 'sc': [sc], 'sod': [sod], 'pot': [pot], 'hemo': [hemo],
        'pcv': [pcv], 'wc': [wc], 'rc': [rc], 'htn': [htn], 'dm': [dm],
        'cad': [cad], 'appet': [appet], 'pe': [pe], 'ane': [ane]
    })
    
    # Ensure columns are in the correct order as used during training
    input_data = input_data[FEATURE_NAMES]
    
    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Make prediction and probability calculation
    prediction_raw = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    # Class 1 is 'ckd' (Chronic Kidney Disease) based on your model's classification encoding
    positive_proba = float(prediction_proba[0][1])

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction_raw[0] == 1:
        st.error(
            f"**PREDICTION: POSITIVE (High Risk of Chronic Kidney Disease)**"
        )
        st.progress(positive_proba, text="Risk Confidence")
        st.metric("Confidence Score", f"{positive_proba:.2%}")
    else:
        st.success(
            f"**PREDICTION: NEGATIVE (Low Risk of Chronic Kidney Disease)**"
        )
        st.progress(positive_proba, text="Risk Confidence")
        st.metric("Confidence Score", f"{positive_proba:.2%}")

st.markdown("---")
st.caption("Model used: Decision tree. **Disclaimer: This is a predictive tool and should not replace professional medical advice.**")
