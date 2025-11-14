import streamlit as st
import joblib
import numpy as np

#load the model
model = joblib.load('best_DecisionTree_model.pkl')

#Title
st.title('Diabetics Prediction App Using Decision Tree Model')

st.markdown('Enter the required features below to get a prediction from your best DT Model')

#Input fields
feature1 = st.number_input('Pregnancies')
feature2 = st.number_input('Glucose')
feature3 = st.number_input('BloodPressure')
feature4 = st.number_input('SkinThickness')
feature5 = st.number_input('Insulin')
feature6 = st.number_input('BMI')
feature7 = st.number_input('Pedigree')
feature8 = st.number_input('Age')

#predict button
if st.button('Predict'):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
    prediction = model.predict(input_data)
    st.success(f'Prediction: {prediction[0]}')

st.markdown('If Prediction is 1, you have Diabetes!')
