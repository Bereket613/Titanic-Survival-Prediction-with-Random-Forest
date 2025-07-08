import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and feature columns
model = joblib.load("titanic_model.pkl")
feature_cols = joblib.load("model_features.pkl")

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    page_icon="🚢"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        color: #1a5276;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sidebar-section {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        text-align: center;
    }
    .survived {
        background-color: #e6f7ee;
        border-left: 5px solid #2ecc71;
    }
    .not-survived {
        background-color: #fde8e8;
        border-left: 5px solid #e74c3c;
    }
    .factor-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-title">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
st.markdown("Enter passenger details to predict survival probability.")

# Sidebar inputs
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("Passenger Details")
    
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Class", [1, 2, 3])
        sex = st.radio("Gender", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
        
    with col2:
        fare = st.slider("Fare ($)", 0, 600, 50)
        sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
        parch = st.slider("Parents/Children", 0, 6, 0)
    
    embarked = st.selectbox("Embarked", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

# Feature engineering
embarked_code = embarked[0]
title_code = "Rare" if title == "Other" else title
family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

# Age and fare grouping
age_group = pd.cut([age], bins=[0, 12, 18, 35, 60, 100], 
                  labels=['Child', 'Teen', 'Adult', 'MidAge', 'Senior'])[0]
fare_group = pd.cut([fare], bins=[-0.01, 10, 50, 100, 600], 
                   labels=['Low', 'Mid', 'High', 'VIP'])[0]

# Create input dataframe
input_data = {
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'SibSp': sibsp,
    'Parch': parch,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Sex': sex,
    'Embarked': embarked_code,
    'Title': title_code,
    'AgeGroup': age_group,
    'FareGroup': fare_group,
    'SocioEconomic': f"{pclass}_{fare_group}"
}

input_df = pd.DataFrame([input_data])

# One-hot encoding
categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'SocioEconomic']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Align with model features
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_cols]

# Prediction button
if st.button("Predict Survival Probability"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][prediction]
        
        # Display results
        result_class = "survived" if prediction == 1 else "not-survived"
        result_emoji = "🎉" if prediction == 1 else "💀"
        result_text = "SURVIVED" if prediction == 1 else "DID NOT SURVIVE"
        
        st.markdown(
            f"""
            <div class="prediction-card {result_class}">
                <h2>{result_emoji} {result_text}</h2>
                <h3>Probability: {probability:.1%}</h3>
                <p>Predicted on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Key factors
        st.subheader("Key Influencing Factors")
        
        factors = []
        if sex == "female":
            factors.append("Female passengers had higher survival rates (74% vs 19% for males)")
        if pclass == 1:
            factors.append(f"First class passengers had {63}% survival rate (vs {47}% for 2nd, {24}% for 3rd)")
        if age <= 12:
            factors.append("Children were prioritized during evacuation")
        if is_alone:
            factors.append("Traveling alone reduced survival chances")
        if fare > 100:
            factors.append("Higher fare tickets correlated with better survival")
        
        for factor in factors:
            st.markdown(f'<div class="factor-card">✓ {factor}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("""
This predictive model was trained on historical Titanic passenger data using a Random Forest classifier.
Model accuracy: ~83% (AUC score).
""")