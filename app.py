import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib
import base64

# Load and prepare data
disease_data = pd.read_csv("disease_data.csv")  # Ensure the file is in the same directory

# Features and target
X = disease_data.drop(columns='target', axis=1)
Y = disease_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model (you can switch to RandomForestClassifier or other complex models)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save model
joblib.dump(model, 'disease_prediction.pkl')

# Load the model
model = joblib.load('disease_prediction.pkl')

# Function to add background image in Streamlit
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            color: black !important;
        }}
        .stTextInput > div {{
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 5px;
        }}
        .stButton > button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
        }}
        .prediction {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background
add_bg_from_local("Disease.jpeg")

# Streamlit UI
st.markdown("<h1 class='title'>Disease Prediction System</h1>", unsafe_allow_html=True)

# Input features from the user
fever = st.selectbox("Fever", options=["Yes", "No"])
cough = st.selectbox("Cough", options=["Yes", "No"])
fatigue = st.selectbox("Fatigue", options=["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath", options=["Yes", "No"])
chest_pain = st.selectbox("Chest Pain", options=["Yes", "No"])
headache = st.selectbox("Headache", options=["Yes", "No"])

# Submit button
submit = st.button("Predict Disease")

# Prediction logic
if submit:
    try:
        # Convert inputs to a numpy array for prediction
        features = [
            1 if fever == "Yes" else 0,
            1 if cough == "Yes" else 0,
            1 if fatigue == "Yes" else 0,
            1 if shortness_of_breath == "Yes" else 0,
            1 if chest_pain == "Yes" else 0,
            1 if headache == "Yes" else 0
        ]
        features_array = np.array(features).reshape(1, -1)
        
        # Predict the disease
        prediction = model.predict(features_array)

        # Show disease prediction
        if prediction[0] == 0:
            st.markdown("<h3 class='prediction'>The person doesn't have a disease.</h3>", unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.markdown("<h3 class='prediction'>Possible Disease: Heart Attack</h3>", unsafe_allow_html=True)
        elif prediction[0] == 2:
            st.markdown("<h3 class='prediction'>Possible Disease: Asthma</h3>", unsafe_allow_html=True)
        elif prediction[0] == 3:
            st.markdown("<h3 class='prediction'>Possible Disease: Pneumonia</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 class='prediction'>Possible Disease: Other conditions</h3>", unsafe_allow_html=True)
    except ValueError:
        st.write("Please provide valid inputs.")

