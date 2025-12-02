import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import xgboost as xgb

# =======================
# APP UI
# =======================

st.set_page_config(page_title="Kidney Disease Prediction", layout="wide")

st.title("🩺 Chronic Kidney Disease Prediction")
st.write("Machine Learning Final Project")
st.write("Predict whether a patient has chronic kidney disease using clinical data.")

# =======================
# LOAD DATA
# =======================

@st.cache_data
def load_data():
    return pd.read_csv("kidney_disease.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =======================
# PREPROCESSING
# =======================

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

# =======================
# FEATURES / TARGET
# =======================

X = df.drop(['classification','id'], axis=1)
y = df['classification']

# =======================
# TRAIN / TEST SPLIT
# =======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# TRAIN MODELS
# =======================

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True)
xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "SVM": svm,
    "XGBoost": xg
}

for m in models.values():
    m.fit(X_train, y_train)

# =======================
# EVALUATION
# =======================

st.subheader("Model Performance")

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label=2)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, acc, f1, auc])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "ROC-AUC"])

st.dataframe(results_df)

# =======================
# CONFUSION MATRICES
# =======================

st.subheader("Confusion Matrices")

fig, axes = plt.subplots(2,2, figsize=(10,8))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
st.pyplot(fig)
plt.clf()

# =======================
# TARGET DISTRIBUTION
# =======================

st.subheader("Kidney Disease Distribution")

fig = plt.figure(figsize=(6,4))
sns.countplot(x='classification', data=df)
plt.title("Target Distribution")
st.pyplot(fig)
plt.clf()

# =======================
# FEATURE IMPORTANCE
# =======================

st.subheader("Feature Importance")

importances = rf.feature_importances_
features = X.columns

fig = plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance")
st.pyplot(fig)
plt.clf()
