import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.title("Breast Cancer Tumor Malignancy Prediction")
st.write("Enter the tumor features to predict malignancy.")

# File upload and reading CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    breast_cancer_dataset = pd.read_csv(uploaded_file)
    breast_cancer_dataset = breast_cancer_dataset.drop('id', axis=1)

    label_encoder = LabelEncoder()
    breast_cancer_dataset['diagnosis'] = label_encoder.fit_transform(breast_cancer_dataset['diagnosis'])

    scaler = StandardScaler()
    X = scaler.fit_transform(breast_cancer_dataset.drop('diagnosis', axis=1))
    y = breast_cancer_dataset['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    values_input = st.text_area("Paste values here (separated by tab or comma or space")
    if st.button("Predict"):
        values_list = values_input.strip().split("\t")
        if len(values_list) != 30:
            # Split using spaces if tab-separated splitting failed
            values_list = values_input.strip().split()
        if len(values_list) != 30:
            # Split using coma if space-separated splitting failed
            values_list = values_input.strip().split(",")
        if len(values_list) == 30:
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, \
            concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, \
            texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, \
            concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, \
            perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, \
            concave_points_worst, symmetry_worst, fractal_dimension_worst = map(float, values_list)

            input_data = np.array([
                [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                 concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                 texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                 concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                 perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                 concave_points_worst, symmetry_worst, fractal_dimension_worst]
            ])
            input_data_scaled = scaler.transform(input_data)
            prediction = log_reg.predict(input_data_scaled)

            if prediction == 0:
                st.write("The tumor is predicted as benign.")
            else:
                st.write("The tumor is predicted as malignant.")
                st.write("The patient potentially has breast cancer.")
        else:
            st.write("Please make sure you have entered 30 values separated by tabs or commas or spaces.")

st.write("---")
st.write("Or manually enter values:")
radius_mean = st.number_input("Enter radius_mean", format="%.2f")
texture_mean = st.number_input("Enter texture_mean", format="%.2f")
perimeter_mean = st.number_input("Enter perimeter_mean", format="%.2f")
area_mean = st.number_input("Enter area_mean", format="%.2f")
smoothness_mean = st.number_input("Enter smoothness_mean", format="%.5f")
compactness_mean = st.number_input("Enter compactness_mean", format="%.5f")
concavity_mean = st.number_input("Enter concavity_mean", format="%.5f")
concave_points_mean = st.number_input("Enter concave points_mean", format="%.5f")
symmetry_mean = st.number_input("Enter symmetry_mean", format="%.4f")
fractal_dimension_mean = st.number_input("Enter fractal_dimension_mean", format="%.5f")
radius_se = st.number_input("Enter radius_se", format="%.4f")
texture_se = st.number_input("Enter texture_se", format="%.4f")
perimeter_se = st.number_input("Enter perimeter_se", format="%.2f")
area_se = st.number_input("Enter area_se", format="%.2f")
smoothness_se = st.number_input("Enter smoothness_se", format="%.6f")
compactness_se = st.number_input("Enter compactness_se", format="%.5f")
concavity_se = st.number_input("Enter concavity_se", format="%.5f")
concave_points_se = st.number_input("Enter concave points_se", format="%.5f")
symmetry_se = st.number_input("Enter symmetry_se", format="%.5f")
fractal_dimension_se = st.number_input("Enter fractal_dimension_se", format="%.5f")
radius_worst = st.number_input("Enter radius_worst", format="%.2f")
texture_worst = st.number_input("Enter texture_worst", format="%.2f")
perimeter_worst = st.number_input("Enter perimeter_worst", format="%.2f")
area_worst = st.number_input("Enter area_worst", format="%.2f")
smoothness_worst = st.number_input("Enter smoothness_worst", format="%.6f")
compactness_worst = st.number_input("Enter compactness_worst", format="%.5f")
concavity_worst = st.number_input("Enter concavity_worst", format="%.5f")
concave_points_worst = st.number_input("Enter concave points_worst", format="%.5f")
symmetry_worst = st.number_input("Enter symmetry_worst", format="%.5f")
fractal_dimension_worst = st.number_input("Enter fractal_dimension_worst", format="%.5f")

if st.button("Predict"):
    input_data = np.array([
        [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
         concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
         texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
         concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
         perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
         concave_points_worst, symmetry_worst, fractal_dimension_worst]
    ])
    input_data_scaled = scaler.transform(input_data)
    prediction = log_reg.predict(input_data_scaled)

    if prediction == 0:
        st.write("The tumor is predicted as benign.")
    else:
        st.write("The tumor is predicted as malignant.")
        st.write("The patient potentially has breast cancer.")
