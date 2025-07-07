import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature columns (exactly what was used during training)
model = joblib.load("titanic_model.pkl")
feature_cols = joblib.load("titanic_model.pkl")  # should be a list of 49 feature names

st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival chances.")

# Sidebar input
st.sidebar.header("🧾 Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
fare = st.sidebar.slider("Fare ($)", 0.0, 500.0, 50.0)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)
embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
title = st.sidebar.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

# Derived features
family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

# AgeGroup binning
age_group = pd.cut([age], bins=[0, 12, 18, 35, 60, 100], 
                   labels=['Child', 'Teen', 'Adult', 'MidAge', 'Senior'])[0]

# FareGroup binning (fixed bins)
fare_group = pd.cut([fare], bins=[-0.01, 10, 50, 100, 600], 
                    labels=['Low', 'Mid', 'High', 'VIP'])[0]

# SocioEconomic
socio = f"{pclass}_{fare_group}"

# Raw input
input_dict = {
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'SibSp': sibsp,
    'Parch': parch,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Sex': sex,
    'Embarked': embarked,
    'Title': title,
    'AgeGroup': age_group,
    'FareGroup': fare_group,
    'SocioEconomic': socio
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns
categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'SocioEconomic']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# 🔒 Ensure exact columns (49) by reindexing
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][prediction]

    if prediction == 1:
        st.success(f"🎉 Survived! (Probability: {prob:.2%})")
    else:
        st.error(f"💀 Did Not Survive (Probability: {prob:.2%})")
