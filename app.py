import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (use the same scaler as during training)
model = joblib.load('models/breast_cancer_prediction_model.joblib')
scaler = joblib.load('models/scaler.joblib')  # Load the scaler used during training

# List of features (same as in your original script)
columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
    "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

# Title of the app
st.title('Breast Cancer Prediction App')

# Sidebar for user inputs
st.sidebar.header('Input Features')

def user_input_features():
    # Creating sliders for each feature
    user_input = {}
    for feature in columns:
        user_input[feature] = st.sidebar.slider(feature, 0.0, 1500.0, 20.0)
    
    input_df = pd.DataFrame(user_input, index=[0])
    return input_df

# Get user input
input_data = user_input_features()

# Standardize the input (use the loaded scaler)
input_data_scaled = scaler.transform(input_data)  # Use the loaded scaler to transform the input data

# Prediction
prediction = model.predict(input_data_scaled)
print(prediction)
# Show result
st.subheader('Prediction Result')

# Fix: Handle single prediction
if prediction[0] == False:
    st.write('The tumor is **Benign**.')
else:
    st.write('The tumor is **Malignant**.')
