
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='MLDP Loan Status Predictor', layout='centered')

st.title('Loan Status Predictor')
st.write('Upload a CSV (same columns as training data, without loan_status) or fill in a single row manually.')

MODEL_PATH = 'outputs/best_model.joblib'
model = joblib.load(MODEL_PATH)

tab1, tab2 = st.tabs(['Upload CSV', 'Single Prediction'])

with tab1:
    file = st.file_uploader('Upload CSV', type=['csv'])
    if file is not None:
        data = pd.read_csv(file)
        st.write('Preview:', data.head())

        preds = model.predict(data)
        out = data.copy()
        out['prediction'] = preds
        out['prediction_label'] = out['prediction'].map({1: 'Approved', 0: 'Rejected'})

        st.success('Predictions generated')
        st.dataframe(out)

        st.download_button(
            label='Download predictions CSV',
            data=out.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )

with tab2:
    st.info('This creates inputs based on columns detected from your dataset.')
    cols = list(getattr(model, 'feature_names_in_', []))

    if len(cols) == 0:
        st.warning('Model does not expose feature_names_in_. Use CSV upload instead.')
    else:
        row = {}
        for c in cols:
            row[c] = st.text_input(c, '')
        if st.button('Predict'):
            df_one = pd.DataFrame([row])
            pred = model.predict(df_one)[0]
            label = 'Approved' if int(pred) == 1 else 'Rejected'
            st.metric('Prediction', label)
