import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
@st.cache_resource
def load_model():
    model_path = 'xg_boost_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload it.")
        return None
    return joblib.load(model_path)

model = load_model()

# Page title
st.title("🚢 Titanic Survival Prediction App")

# User input
st.header("Enter Passenger Details:")
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("No. of Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("No. of Parents/Children Aboard", min_value=0, step=1)
fare = st.number_input("Passenger Fare", min_value=0.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Derived features
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Prepare input
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == "male" else 0],  # Encode male as 1, female as 0
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],
    'FamilySize': [family_size],
    'IsAlone': [is_alone]
})

# Encode Embarked column manually
input_df['Embarked'] = input_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Predict
if st.button("Predict Survival"):
    if model is not None:
        try:
            prediction = model.predict(input_df)[0]
            outcome = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
            st.subheader(f"Prediction: {outcome}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
