🍷 Alcohol Category Detection using Machine Learning
🚀 End-to-End ML Project | XGBoost | Streamlit | Model Deployment

This project demonstrates a complete machine learning pipeline — from data preprocessing and model training to deployment as an interactive web application.
The system predicts whether an alcohol sample belongs to a Good or Not Good quality category based on its physicochemical properties.

🎯 Goal: Showcase real-world ML deployment skills, not just model training.

🔍 Problem Statement

Manual evaluation of alcohol quality can be subjective and time-consuming.
This project automates quality classification using a supervised machine learning approach, enabling fast and consistent predictions.

🧠 Solution Overview

Trained a binary classification model using XGBoost

Applied feature scaling with StandardScaler

Persisted model and scaler using joblib

Deployed the model using Streamlit for real-time predictions

🛠️ Tech Stack
Category	Tools
Language	Python
ML Model	XGBoost
ML Libraries	Scikit-learn, Pandas
Model Persistence	Joblib
Web Framework	Streamlit
📊 Input Features

Fixed acidity

Volatile acidity

Citric acid

Residual sugar

Chlorides

Free sulfur dioxide

Total sulfur dioxide

Density

pH

Sulphates

Alcohol

⚙️ ML Pipeline

Data preprocessing and feature selection

Standardization using StandardScaler

Model training with XGBoost

Hyperparameter optimization (best parameters saved)

Model serialization for deployment

Real-time inference via Streamlit UI

🖥️ Application Demo

Users input chemical attributes and instantly receive a predicted alcohol quality category.

✔️ Fast inference
✔️ Clean UI
✔️ Production-ready structure

📁 Project Structure
├── app.py                       # Streamlit application
├── xgb_classifier_best_param.pkl # Trained XGBoost model
├── xgb_scaler.pkl               # Feature scaler
├── requirements.txt             # Dependencies
└── README.md

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py

🎯 What This Project Demonstrates (For Recruiters)

✅ End-to-end ML workflow
✅ Model deployment skills
✅ Feature preprocessing awareness
✅ Handling ML dependencies (XGBoost)
✅ Real-world inference pipeline
✅ Clean, modular, maintainable code

📌 Key Learnings

Trained models remain dependent on their originating libraries (XGBoost)

Environment consistency is critical for ML deployment

Streamlit enables rapid ML prototyping and delivery

🚀 Future Improvements

Probability-based predictions

Model explainability (SHAP)

Multi-class alcohol quality classification

Cloud deployment (Streamlit Cloud / Docker)
