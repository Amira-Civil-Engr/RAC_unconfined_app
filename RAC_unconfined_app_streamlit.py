
import streamlit as st
import pandas as pd
import joblib

# Function to load the unconfined model and scaler
def load_model_scaler_unconfined():
    scaler = joblib.load("scaler_CS.pkl")
    model = joblib.load("best_model_CS.pkl")
    return model, scaler

# Function to predict using CSV file for unconfined RAC
def predict_from_csv_unconfined(df):
    required_columns = ['C', 'W', 'NFA', 'NCA', 'RFA', 'RCA', 'SF', 'FA', 'Age']
    model, scaler = load_model_scaler_unconfined()
    X_scaled = scaler.transform(df[required_columns])
    predictions = model.predict(X_scaled)
    df['Predicted CS (MPa)'] = predictions
    return df

# Function to handle manual input predictions for unconfined RAC
def predict_from_manual_input_unconfined(inputs):
    input_data = pd.DataFrame([inputs])
    model, scaler = load_model_scaler_unconfined()
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)
    st.write(f"**Predicted Compressive Strength (CS): {prediction[0]:.2f} MPa**")

# Main App Interface for Unconfined RAC
st.title("Unconfined RAC Prediction App")
st.markdown("Predict the compressive strength of unconfined Recycled Aggregate Concrete (RAC).")

# Upload CSV file or enter data manually
st.markdown("### Input Data")
upload_method = st.radio("Choose how to input data:", ('Upload CSV File', 'Enter Manually'))

# For CSV input
if upload_method == 'Upload CSV File':
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            result_df = predict_from_csv_unconfined(df)
            st.dataframe(result_df)

            # Allow the user to download the prediction results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='prediction_results.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# For manual input
elif upload_method == 'Enter Manually':
    inputs = {
        'C': st.number_input('Cement Content (C) [kg/m³]', min_value=0.0),
        'W': st.number_input('Water Content (W) [kg/m³]', min_value=0.0),
        'NFA': st.number_input('Natural Fine Aggregate (NFA) [kg/m³]', min_value=0.0),
        'NCA': st.number_input('Natural Coarse Aggregate (NCA) [kg/m³]', min_value=0.0),
        'RFA': st.number_input('Recycled Fine Aggregate (RFA) [kg/m³]', min_value=0.0),
        'RCA': st.number_input('Recycled Coarse Aggregate (RCA) [kg/m³]', min_value=0.0),
        'SF': st.number_input('Silica Fume Content (SF) [kg/m³]', min_value=0.0),
        'FA': st.number_input('Fly Ash Content (FA) [kg/m³]', min_value=0.0),
        'Age': st.number_input('Age (days)', min_value=0.0)
    }
    if st.button("Predict"):
        predict_from_manual_input_unconfined(inputs)


# Footer
st.markdown("---")
st.markdown("© 2024 (Amira Ahmed, Wu Jin, Mosaad Ali ). All rights reserved.")
st.markdown("Developed by Amira Ahmed. Contact: amira672012@yahoo.com")

