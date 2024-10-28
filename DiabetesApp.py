import numpy as np 
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import pickle

st.markdown("<h1 style='text-align: center;'>Diabetes Detection App</h1>", unsafe_allow_html=True)

image=Image.open("dia2.jpeg")

st.image(image)

st.subheader("Enter your input data:")

def user_input_features():
    pregnancies=st.slider("Pregnancies",0,20,0, help="Number of times pregnant")
    glucose=st.slider("Glucose",0.00,300.00,0.00, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    bloodPressure=st.slider("BloodPressure",0.00,200.00,0.00, help="Diastolic blood pressure (mm Hg)")
    skinThickness=st.slider("SkinThickness",0.00,100.00,0.00, help="Triceps skin fold thickness (mm)")
    insulin=st.slider("Insulin",0.00,400.00,0.00, help=" 2-Hour serum insulin (mu U/ml)")
    bmi=st.slider("BMI",0.00,80.00,0.00, help="Body mass index (weight in kg/(height in m)^2)")
    diabetesPedigreeFunction=st.slider("DiabetesPedigreeFunction",0.00,3.00,0.00, help="A function which scores likelihood of diabetes based on family history.")
    age=st.slider("Age",0,100,0, help="Age (years)")

    input_dict={
        "Pregnancies":pregnancies,
        "Glucose":glucose,
        "BloodPressure":bloodPressure,
        "SkinThickness":skinThickness,
        "Insulin":insulin,
        "BMI":bmi,
        "DiabetesPedigreeFunction":diabetesPedigreeFunction,
        "Age":age
    }

    features=pd.DataFrame(input_dict,index=['User_Input_Values'])
    return features


ui=user_input_features()

st.write(ui)

st.subheader("Prediction:")

min_max_values=joblib.load('min_max_vals.joblib')

for col in ui.columns:
    min_val,max_val=min_max_values[col]
    ui[col]=(ui[col]-min_val)/(max_val-min_val)

with open('DiaModel.pkl','rb') as f:
    model_loaded=pickle.load(f)

prediction=model_loaded.predict(ui)

st.write(prediction)

st.info("1 - Positive, 0 - Negative")
