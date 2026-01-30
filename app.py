import streamlit as st
import pandas as pd
import joblib

label_map = {
    0: "Not Good",
    1: "Good"
}

FEATURE_ORDER = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

@st.cache_resource
def load_model():
    return joblib.load("xgb_classifier_best_param.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("xgb_scaler.pkl")

def preprocessing_input_data(data):
    df = pd.DataFrame([data])
    df = df[FEATURE_ORDER]   # ensure correct column order
    scaler = load_scaler()
    return scaler.transform(df)

def predict_data(data):
    model = load_model()
    processed_data = preprocessing_input_data(data)
    prediction = model.predict(processed_data)
    return prediction[0]

def main():
    st.title("🍷 Alcohol Category Detection")

    fs  = st.number_input("Fixed Acidity", min_value=0.0)
    va  = st.number_input("Volatile Acidity", min_value=0.0)
    ca  = st.number_input("Citric Acid", min_value=0.0)
    rs  = st.number_input("Residual Sugar", min_value=0.0)
    ch  = st.number_input("Chlorides", min_value=0.0)
    fsd = st.number_input("Free Sulfur Dioxide", min_value=0.0)
    tsd = st.number_input("Total Sulfur Dioxide", min_value=0.0)
    d   = st.number_input("Density", min_value=0.0)
    ph  = st.number_input("pH", min_value=0.0)
    ss  = st.number_input("Sulphates", min_value=0.0)
    a   = st.number_input("Alcohol", min_value=0.0)

    if st.button("Predict Alcohol Category"):
        user_data = {
            'fixed acidity': fs,
            'volatile acidity': va,
            'citric acid': ca,
            'residual sugar': rs,
            'chlorides': ch,
            'free sulfur dioxide': fsd,
            'total sulfur dioxide': tsd,
            'density': d,
            'pH': ph,
            'sulphates': ss,
            'alcohol': a
        }

        prediction = predict_data(user_data)
        alc_cat = label_map[prediction]
        st.success(f"✅ Predicted Category: **{alc_cat}**")

if __name__ == "__main__":
    main()
