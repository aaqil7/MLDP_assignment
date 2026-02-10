import os, json, joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE_DIR, 'outputs')

model = joblib.load(os.path.join(OUTPUTS, 'best_model.joblib'))
scaler = joblib.load(os.path.join(OUTPUTS, 'scaler.joblib'))

with open(os.path.join(OUTPUTS, 'train_columns.json'), 'r') as f:
    TRAIN_COLS = json.load(f)

st.title('Loan Status Prediction')

# IMPORTANT: use the same feature names as your dataset
income_annum = st.number_input('income_annum', min_value=0, value=120000)
no_of_dependents = st.number_input('no_of_dependents', min_value=0, value=3)
loan_amount = st.number_input('loan_amount', min_value=0, value=200000)
loan_term = st.number_input('loan_term', min_value=0, value=360)
cibil_score = st.number_input('cibil_score', min_value=0, max_value=900, value=750)

residential_assets_value = st.number_input('residential_assets_value', min_value=0, value=0)
commercial_assets_value = st.number_input('commercial_assets_value', min_value=0, value=0)
luxury_assets_value = st.number_input('luxury_assets_value', min_value=0, value=0)
bank_asset_value = st.number_input('bank_asset_value', min_value=0, value=0)

education = st.selectbox('education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('self_employed', ['Yes', 'No'])

if st.button('Predict'):
    df_input = pd.DataFrame([{
        'income_annum': income_annum,
        'no_of_dependents': no_of_dependents,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'education': education,
        'self_employed': self_employed
    }])

    # SAME as notebook
    df_enc = pd.get_dummies(df_input, drop_first=True)
    df_enc = df_enc.reindex(columns=TRAIN_COLS, fill_value=0)

    X_scaled = scaler.transform(df_enc)
    pred = model.predict(X_scaled)[0]

    st.success(f'Prediction: {pred}')
