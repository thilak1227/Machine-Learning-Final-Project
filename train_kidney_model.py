import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# --- 1. CONFIGURATION ---
MODEL_FILENAME = 'xgboost_kidney_model.pkl'
SCALER_FILENAME = 'kidney_scaler.pkl'

# --- 2. DATA LOADING AND PREPROCESSING ---
def load_and_preprocess_data():
    # Loading the data
    df = pd.read_csv("kidney_disease.csv")
    
    # Categorical Columns (from your original script)
    categorical_cols = [
        'rbc','pc','pcc','ba','htn','dm','cad',
        'appet','pe','ane','classification'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # Convert numeric text columns
    for col in ['pcv','wc','rc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing numeric values
    num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Features (X) and Target (y)
    X = df.drop(['classification','id'], axis=1)
    y = df['classification']
    
    return X, y

# --- 3. MODEL TRAINING AND SAVING ---
def train_and_save_model():
    X, y = load_and_preprocess_data()
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train XGBoost model (the best performer based on your matrices)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(model, file)
        
    # Save the fitted scaler (CRITICAL for preprocessing new user input)
    with open(SCALER_FILENAME, 'wb') as file:
        pickle.dump(scaler, file)
        
    print(f"Model saved as {MODEL_FILENAME}")
    print(f"Scaler saved as {SCALER_FILENAME}")

    # Optionally print performance on test set
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import accuracy_score
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == '__main__':
    train_and_save_model()