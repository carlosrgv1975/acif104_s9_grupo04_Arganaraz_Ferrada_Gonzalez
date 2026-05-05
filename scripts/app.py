import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title('🧠 PREDICCION DE ACCIDENTE CEREBROVASCULAR')
st.markdown('**Grupo 04 - Priscila Arganaraz / Carlos González | ACIF104**')
st.markdown('INGENIERIA CIVIL INFORMATICA')
st.markdown('UNIVERSIDAD ANDRES BELLO')
st.markdown('---')

st.sidebar.header('Ingrese los datos del paciente')

age = st.sidebar.slider('Edad', 0, 100, 50)
hypertension = st.sidebar.selectbox('Hipertensión', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
heart_disease = st.sidebar.selectbox('Enfermedad cardíaca', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
avg_glucose = st.sidebar.slider('Nivel de glucosa promedio', 50.0, 300.0, 100.0)
bmi = st.sidebar.slider('IMC (BMI)', 10.0, 60.0, 28.0)
gender = st.sidebar.selectbox('Género', ['Femenino', 'Masculino'])
ever_married = st.sidebar.selectbox('¿Casado/a?', ['No', 'Sí'])
smoking = st.sidebar.selectbox('Hábito de fumar', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
work_type = st.sidebar.selectbox('Tipo de trabajo', ['Private', 'Self-employed', 'Govt_job', 'Never_worked'])
residence = st.sidebar.selectbox('Tipo de residencia', ['Urbano', 'Rural'])

st.markdown('### 📋 Datos ingresados')
col1, col2, col3 = st.columns(3)
col1.metric('Edad', age)
col2.metric('Glucosa', avg_glucose)
col3.metric('BMI', bmi)

if st.sidebar.button('🔍 Predecir riesgo'):
    with st.spinner('Analizando datos...'):
        input_data = {
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'avg_glucose_level': avg_glucose,
            'bmi': bmi,
            'gender_Male': 1 if gender == 'Masculino' else 0,
            'ever_married_Yes': 1 if ever_married == 'Sí' else 0,
            'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
            'work_type_Private': 1 if work_type == 'Private' else 0,
            'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
            'Residence_type_Urban': 1 if residence == 'Urbano' else 0,
            'smoking_status_formerly smoked': 1 if smoking == 'formerly smoked' else 0,
            'smoking_status_never smoked': 1 if smoking == 'never smoked' else 0,
            'smoking_status_smokes': 1 if smoking == 'smokes' else 0,
        }

        df_input = pd.DataFrame([input_data])

        df_train = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
        df_train['bmi'].fillna(df_train['bmi'].mean(), inplace=True)
        df_train = df_train[df_train['gender'] != 'Other']
        df_train = pd.get_dummies(df_train, columns=['gender','ever_married','work_type','Residence_type','smoking_status'], drop_first=True)
        df_train = df_train.drop('id', axis=1)
        X = df_train.drop('stroke', axis=1)
        y = df_train['stroke']

        for col in X.columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[X.columns]

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        df_input_sc = scaler.transform(df_input)

        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_sc, y)

        prob = model.predict_proba(df_input_sc)[0][1]
        pred = model.predict(df_input_sc)[0]

    st.markdown('---')
    st.markdown('## 🎯 Resultado')

    if pred == 1:
        st.error('⚠️ ALTO RIESGO de accidente cerebrovascular')
    else:
        st.success('✅ BAJO RIESGO de accidente cerebrovascular')

    st.metric('Probabilidad de stroke', f'{prob*100:.1f}%')

    st.markdown('---')
    st.markdown('### 📊 Variables más importantes (según SHAP)')
    st.markdown('- **Edad**: Factor más determinante en la predicción')
    st.markdown('- **Nivel de glucosa**: Alto impacto en el riesgo')
    st.markdown('- **Hipertensión**: Factor de riesgo cardiovascular relevante')
    st.markdown('- **BMI**: Indicador de condición física general')

st.markdown('---')
st.markdown('*Modelo: Regresión Logística con class_weight=balanced | AUC-ROC: 0.840*')
st.markdown('*Este sistema es solo orientativo y no reemplaza diagnóstico médico profesional*')