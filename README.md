# AI-Based early prediction of Chronic Kidney Disease(CDK) using machine learning models.
# About the Project
<div align="justify">
This project extends my degree coursework by developing and evaluating machine learning models to predict Chronic KIdney Disease using the Chronic KIdney Disease dataset from the UCI repository. It emphasizes the technical aspects of model selection, training, and performance evaluation, highlighting the effectiveness of algorithms such as Logistic Regression, Support Vector Machine, Random Forest, and XGBoost in clinical prediction tasks.
</div>

# Background Overview
<div align="justify">
Chronic Kidney Disease (CKD) is becoming increasingly common worldwide, and many individuals only learn about their condition when the disease has already progressed to a more serious stage. This happens mainly because early CKD usually does not show clear symptoms, making it difficult for patients and healthcare providers to detect it in time.
The purpose of this project is to build a machine-learning model that can assist in identifying early signs of CKD using simple clinical indicators such as blood pressure, blood sugar, creatinine level, hemoglobin, and urine protein. By analyzing these health measurements, the system can estimate whether a patient may be at risk and help guide further medical evaluation.
We trained and compared multiple machine-learning models to determine which one performs best. The final system is designed to be used through a simple interface, where users can input patient data and receive an instant prediction. This approach can support healthcare professionals in making faster and more informed decisions, potentially improving patient outcomes.
</div>

# Problem Statement
<div align="justify">
Chronic Kidney Disease (CKD) is a growing global health problem, and many patients only discover the condition when it has already reached an advanced stage. One of the main challenges is that early CKD often shows very mild or no symptoms at all. Because of this, many individuals live with kidney damage for years without receiving proper medical care, which increases the risk of kidney failure and long-term complications. Although doctors rely on medical indicators such as creatinine levels, blood pressure, hemoglobin, and urine protein to diagnose CKD, interpreting these values manually can be challenging especially when results are borderline or when clinics are busy. Early signs may be missed, which delays treatment.
This situation highlights the need for a system that can analyze patient data quickly and consistently. Machine-learning models can recognize patterns in clinical data that may not be obvious during routine checkups. By detecting early risks, these models can support healthcare providers in offering timely intervention and reducing long-term health consequences
This creates a clear need for a system that can analyze patient data quickly and support early decision-making. By using machine learning models trained on real patient data, it becomes possible to detect patterns that may not be obvious during routine checkups. An AI-based prediction tool can help healthcare providers identify high-risk cases earlier, improve treatment planning, and reduce the chances of long-term kidney complications.
</div>

# Project Objectives: Chronic Kidney Disease Prediction

This project focuses on leveraging machine learning to enhance the early detection and risk assessment of Chronic Kidney Disease (CKD) using clinical patient data.

## Key Objectives

The specific goals of this project are:

1.  **Data Preparation and Quality:** To prepare and preprocess the clinical CKD data, ensuring **quality, consistency, and numerical suitability** for machine learning algorithms.

2.  **Model Implementation and Comparison:** To implement and train multiple robust classification models, including **Logistic Regression, Random Forest, SVM, and XGBoost**, to compare their predictive performance.

3.  **Rigorous Performance Evaluation:** To evaluate model efficacy using a comprehensive set of metrics: **Accuracy, F1-score, ROC-AUC, and Confusion Matrices**.

4.  **Model Optimization and Generalization:** To improve model stability and generalizability through techniques such as **hyperparameter tuning and cross-validation**.

5.  **Deployment as Decision Support Tool:** To save the best-performing model and deploy it via **Streamlit**, creating an accessible, interactive tool for instant CKD risk prediction.

## Project Vision

By demonstrating high-accuracy prediction capabilities, this project seeks to highlight the potential of machine learning as a **data-driven decision-support tool**, complementing traditional diagnostics in the advancement of healthcare.

# Built With
*Google Colab*

# Dataset Source
<div align="justify">
The dataset used in this project is the Chronic Kidney Disease (CKD) Dataset obtained from Kaggle, a well-known platform for open-source datasets.
It contains 400 patient records and 25 clinical attributes, along with a label indicating whether the patient has CKD or not.

Dataset link: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

</div>

# Project Methodology: Chronic Kidney Disease Prediction

## 1. Data Collection
The Chronic Kidney Disease (CKD) dataset was sourced from **Kaggle** and imported for initial analysis and development within a Jupyter Notebook environment.

---

## 2. Data Cleaning & Preprocessing
The raw data underwent extensive cleaning to prepare it for model training:

* **Handling Missing Values:** Missing categorical values were imputed, and missing numerical features were filled using the **median** of the respective column.
* **Encoding Categorical Data:** All categorical attributes (e.g., `rbc`, `htn`) were converted into a numerical format using **Label Encoding**.
* **Scaling/Normalization:** Selected numerical features were standardized using a **StandardScaler** to prevent features with larger scales from dominating the model training process.
* **Data Integrity:** Duplicate records were removed, and the dataset was checked for outliers.

---

## 3. Feature Selection
Relevant clinical attributes were carefully selected to build the model based on domain knowledge and correlation analysis. Key attributes used include:
* Blood pressure
* Specific gravity
* Hemoglobin
* Serum Creatinine

---

## 4. Model Development
Four machine learning classification algorithms were trained and their performance was rigorously compared:
* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **Random Forest** 

> **Goal:** The primary objective was to achieve a predictive accuracy of **80–85% or higher** on the test dataset.

---

## 5. Model Evaluation
Each model was assessed using a comprehensive set of metrics to determine its overall effectiveness and reliability:

* **Quantitative Metrics:** Accuracy, Precision, Recall, and F1-score.
* **Visual Assessment:** **Confusion Matrices** were generated for all models to visualize classification errors and analyze True Positive, True Negative, False Positive, and False Negative rates.

---

## 6. Deployment
The best-performing model (the **Random Forest**) was saved and deployed to create an accessible end-user tool:

* The final trained model and the fitted data scaler were saved using the Python `pickle` library.
* The model was deployed as an interactive web application using **Streamlit**, allowing users to enter patient clinical data and receive an instant CKD risk prediction.
# Result and Impact
<div align="justify">
After training the four models, the accuracy scores were compared. The Random Forest model achieved the highest accuracy.
Key findings include:
	* Random Forest performed best due to its ability to handle non-linear relationships.
	*	Logistic Regression was stable but less accurate.
	*	KNN was sensitive to feature scaling.
	*	Decision Tree worked well but was more prone to overfitting.

</div>

# Challenges and Solution
* Class Imbalance in Dataset: Addressed using stratified train–test split and evaluation metrics like F1 score and ROC AUC instead of accuracy, ensuring balanced detection of both classes.
* Categorical Variables and Scaling: Handled with label and one-hot encoding to preserve clinical meaning, while StandardScaler normalized numerical features for fair model training.
* Overfitting Risk: Mitigated through cross-validation and hyperparameter tuning, allowing ensemble models such as Random Forest and XGBoost to generalize effectively.
* Model Interpretability: Enhanced by feature importance analysis, highlighting medically relevant predictors and bridging the gap between accuracy and clinical trust.

# Model Development
https://colab.research.google.com/drive/1SanBPlf5cLPsHZgSKKqGQK1l2dYzJ40u?authuser=1

# How to Use
How to Use
To test the application, click this link:
https://machine-learning-final-project-kideny.streamlit.app/

# Model Evaluation

=== Final Model Comparison ===
                     Train Accuracy  Test Accuracy  CV Mean Accuracy  \
Tuned Decision Tree          0.9875         0.9625            0.9625   
Tuned Random Forest          0.9906         0.9750            0.9906  


# Acknowledgement
<div align="justify">
We would like to express our gratitude to our lecturer, Sir Nazmirul Izzad Bin Nassir, for his invaluable guidance and support throughout this project. We also extend our appreciation to our team members for their collaboration and dedication in completing this work. The dataset used in this study was obtained from Kaggle, which served as the foundation for our analysis and experimentation.
</div>

## Team Members
* Thilak A/L Kamalish Kumar (202309010104, BIT)
* SIJAN MD SHAHDAT HOSSAIN (202501010545, BCSSE)
* Daabosh Ahmed Mahmood Ali (202501010453, BCSSE)
* Natasha Najwa Abdullah (202409010558, BCSSE)

## Course Information
* Subject Code: BIT4333
* Subject Name: Introduction to Machine Learning
* Submission Date: 4th December 2025


