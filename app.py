import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import base64

disease_data = pd.read_csv("disease_data.csv")

def preprocess_data(df):
    for col in ['fever','cough','fatigue','shortness_of_breath','chest_pain','headache']:
        df[col] = df[col].replace([0,'0',np.nan], 'No')
        df[col] = df[col].map({'Yes':1, 'No':0})
    if 'target' in df.columns:
        target_map = {'No Disease':0, 'Heart Attack':1, 'Asthma':2, 'Pneumonia':3}
        df['target'] = df['target'].map(target_map)
    df['fever_cough'] = df['fever'] * df['cough']
    df['fatigue_chestpain'] = df['fatigue'] * df['chest_pain']
    df['sob_chestpain'] = df['shortness_of_breath'] * df['chest_pain']
    return df

disease_data = preprocess_data(disease_data)

X = disease_data.drop(columns='target', axis=1)
y = disease_data['target']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

joblib.dump(model, 'disease_prediction.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')

st.set_page_config(page_title="Disease Prediction System", layout="centered")

model = joblib.load('disease_prediction.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
    <style>
    /* Remove top padding/margin in Streamlit reliably */
    header {{visibility: hidden;}}
    .appview-container .main {{padding-top: 0px !important;}}
    
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white !important;
        font-weight: bold !important;
    }}
    .title, .prediction {{
        color: white !important;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }}
    .stDataFrame div, .stDataFrame th, .stDataFrame td {{
        color: white !important;
        font-weight: bold !important;
    }}
    label {{
        color: white !important;
        font-weight: bold !important;
    }}
    div.stButton > button {{
        background-color: #ADD8E6;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 40px;
        width: 220px;
    }}
    div.stButton > button:hover {{
        background-color: #1E90FF;
        color: white;
    }}
    .stFileUploader button {{
        background-color: #ADD8E6 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        height: 40px !important;
        width: 220px !important;
    }}
    .stFileUploader button:hover {{
        background-color: #1E90FF !important;
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)


add_bg_from_local("img.jpg")

st.markdown("<h1 class='title'>Disease Prediction System</h1>", unsafe_allow_html=True)

st.subheader("Predict for a Single Person")
fever = st.selectbox("Fever", options=["Yes", "No"])
cough = st.selectbox("Cough", options=["Yes", "No"])
fatigue = st.selectbox("Fatigue", options=["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath", options=["Yes", "No"])
chest_pain = st.selectbox("Chest Pain", options=["Yes", "No"])
headache = st.selectbox("Headache", options=["Yes", "No"])

submit_single = st.button("Predict Single Person")

if submit_single:
    user_input = pd.DataFrame([[fever,cough,fatigue,shortness_of_breath,chest_pain,headache]],
                              columns=['fever','cough','fatigue','shortness_of_breath','chest_pain','headache'])
    user_input = preprocess_data(user_input)
    user_poly = poly.transform(user_input)
    user_scaled = scaler.transform(user_poly)
    prediction = model.predict(user_scaled)[0]
    disease_mapping = {0: "No Disease", 1: "Heart Attack", 2: "Asthma", 3: "Pneumonia"}
    color = "#2ECC71" if prediction==0 else "#E74C3C"
    st.markdown(f"<h3 class='prediction' style='color:{color}'>Prediction: {disease_mapping[prediction]}</h3>", unsafe_allow_html=True)

st.subheader("Predict for Multiple People (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV with columns: fever,cough,fatigue,shortness_of_breath,chest_pain,headache", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    batch_data_processed = preprocess_data(batch_data.copy())
    batch_poly = poly.transform(batch_data_processed)
    batch_scaled = scaler.transform(batch_poly)
    batch_pred = model.predict(batch_scaled)
    disease_mapping = {0:"No Disease", 1:"Heart Attack", 2:"Asthma", 3:"Pneumonia"}
    batch_data['Prediction'] = [disease_mapping[i] for i in batch_pred]
    st.write("Predictions:")
    st.dataframe(batch_data)
