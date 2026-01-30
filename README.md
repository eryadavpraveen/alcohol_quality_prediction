# 🍷 Alcohol Category Detection using Machine Learning

### 🚀 End-to-End ML Project | XGBoost | Streamlit | Model Deployment

This project demonstrates a **complete machine learning pipeline** — from data preprocessing and model training to deployment as an **interactive web application**.  
The system predicts whether an alcohol sample belongs to a **Good** or **Not Good** quality category based on its physicochemical properties.

🎯 **Goal:** Showcase real-world ML deployment skills, not just model training.

---

## 🔍 Problem Statement

Manual evaluation of alcohol quality can be **subjective and time-consuming**.  
This project automates quality classification using a **supervised machine learning approach**, enabling fast and consistent predictions.

---

## 🧠 Solution Overview

- Trained a **binary classification model** using **XGBoost**
- Applied feature scaling with **StandardScaler**
- Persisted model and scaler using **joblib**
- Deployed the model using **Streamlit** for real-time predictions

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| ML Model | XGBoost |
| ML Libraries | Scikit-learn, Pandas |
| Model Persistence | Joblib |
| Web Framework | Streamlit |

---

## 📊 Input Features

- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

---

## ⚙️ Machine Learning Pipeline

1. Data preprocessing and feature selection  
2. Feature scaling using `StandardScaler`  
3. Model training with XGBoost  
4. Hyperparameter tuning and best model selection  
5. Model serialization using joblib  
6. Real-time inference through Streamlit UI  

---

## 🖥️ Application Overview

Users enter physicochemical attributes of alcohol and receive an instant **quality category prediction**.

✔️ Fast inference  
✔️ Clean UI  
✔️ Deployment-ready structure  

---

## 📁 Project Structure
#├── app.py                       # Streamlit application<br>
#├── xgb_classifier_best_param.pkl # Trained XGBoost model<br>
#├── xgb_scaler.pkl               # Feature scaler<br>
#├── requirements.txt             # Dependencies<br>
#└── README.md<br>


---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

