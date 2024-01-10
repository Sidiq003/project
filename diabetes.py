import pickle
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt

diabetes_model = pickle.load(open('diabetes.sav', 'rb'))

st.title("Prediksi Diabetes Dengan KNN")

Pregnancies = st.slider ("Nilai Pregnancies/ Kehamilan", 0, 17)
Glucose = st.number_input ("Nilai Glucose/ Glukosa", step=1.0)
BloodPressure = st.number_input ("Nilai Blood Pressure/ Tekanan Darah", step=1.0)
SkinThickness = st.number_input ("Nilai Skin Thickness/ Ketebalan Kulit", step=1.0)
Insulin = st.number_input ("Nilai Insulin", step=1.0)
BMI = st.number_input ("Nilai BMI", step=1.0)
DiabetesPedigreeFunction = st.number_input ("Nilai Diabetes Pedigree Function/ Fungsi Silsilah Diabetes", step=1.0)
Age = st.number_input ("Nilai Age/ Usia", step=1.0)

diab_diagnosis = ''

if st.button('TESTING'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age]])
        if diab_prediction[0] == 1:
                diab_diagnosis = "Pasien Terkena Diabetes"
        else:
                diab_diagnosis = "Pasien Tidak Terkena Diabetes"
        st.success(diab_diagnosis)

st.subheader("")
st.subheader("DataFrame Yang Digunakan")
df = pd.read_csv("diabetes.csv")
st.write("Data:")
st.write(df)

diabetes = df[df['Outcome'] == 1]
non_diabetes = df[df['Outcome'] == 0]

st.subheader("")
st.subheader("Visualisasi Data")
plt.figure(figsize=(8, 6))
plt.scatter(diabetes['Age'], diabetes['Glucose'], color='red', label='Diabetes', alpha=0.4)
plt.scatter(non_diabetes['Age'], non_diabetes['Glucose'], color='green', label='Non-Diabetes', alpha=0.4)
plt.xlabel('Usia (Age)')
plt.ylabel('Kadar Glukosa (Glucose)')
plt.title('Usia vs Kadar Glukosa')
plt.legend()

st.pyplot(plt)

st.subheader("")
st.subheader("Kelompok Data Mining")
st.write("1. ..."
         "\n2. ..."
         "\n3. ..."
         "\n4. ...")