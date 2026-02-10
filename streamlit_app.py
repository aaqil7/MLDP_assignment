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
st.caption('Enter applicant details. Example values are shown as hints.')

income_annum = st.number_input('income_annum', min_value=0, value=0, help='Example: 120000')
no_of_dependents = st.number_input('no_of_dependents', min_value=0, value=0, help='Example: 3')
loan_amount = st.number_input('loan_amount', min_value=0, value=0, help='Example: 200000')
loan_term = st.number_input('loan_term', min_value=0, value=0, help='Example: 360')
cibil_score = st.number_input('cibil_score', min_value=0, max_value=900, value=0, help='Example: 750')

residential_assets_value = st.number_input('residential_assets_value', min_value=0, value=0, help='Example: 200000')
commercial_assets_value = st.number_input('commercial_assets_value', min_value=0, value=0, help='Example: 100000')
luxury_assets_value = st.number_input('luxury_assets_value', min_value=0, value=0, help='Example: 50000')
bank_asset_value = st.number_input('bank_asset_value', min_value=0, value=0, help='Example: 30000')

education = st.selectbox('education', ['Select an option', 'Graduate', 'Not Graduate'])
self_employed = st.selectbox('self_employed', ['Select an option', 'Yes', 'No'])

if st.button('Predict'):
    if education == 'Select an option' or self_employed == 'Select an option':
        st.warning('Please select values for education and self_employed.')
        st.stop()

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

    df_enc = pd.get_dummies(df_input, drop_first=True)
    df_enc = df_enc.reindex(columns=TRAIN_COLS, fill_value=0)

    X_scaled = scaler.transform(df_enc)
    pred = model.predict(X_scaled)[0]

label = 'Yes' if pred == 1 else 'No'
st.success(f'Loan Approved: {label}')
