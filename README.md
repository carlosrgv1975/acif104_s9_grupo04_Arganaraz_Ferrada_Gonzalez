# Prediccion de Accidente Cerebrovascular mediante Machine Learning
## ACIF104 — Fase 3 | Grupo 04 | Universidad Andres Bello

**Integrantes:** Priscila Arganaraz · Nicolas Ferrada · Carlos Gonzalez  
**Carrera:** Ingenieria Civil Informatica  
**Curso:** ACIF104 Aprendizaje de Maquinas  

---

## DESCRIPCION DEL PROYECTO

Sistema de prediccion de accidentes cerebrovasculares (stroke) desarrollado en tres fases utilizando tecnicas de Machine Learning y Deep Learning. El proyecto incluye analisis exploratorio de datos, comparacion de modelos, explicabilidad con SHAP y una aplicacion web interactiva para prediccion en tiempo real.


## ESTRUCTURA DEL REPOSITORIO
```
acif104_s9_grupo04_Arganaraz_Ferrada_Gonzalez/
├── datasets/
│   └── healthcare-dataset-stroke-data.csv   # Dataset original de Kaggle
├── notebooks/
│   └── acif104_s9_grupo04_Arganaraz_Ferrada_Gonzalez.ipynb  # Notebook principal
├── scripts/
│   └── app.py                               # Aplicacion Streamlit (frontend + backend)
├── modelos_entrenados/
│   ├── modelo_stroke.pkl                    # Modelo Regresion Logistica serializado
│   └── scaler_stroke.pkl                    # StandardScaler serializado
├── imagenes/
│   ├── eda_distribuciones.png
│   ├── eda_correlacion.png
│   ├── eda_categoricas.png
│   ├── particion_conjuntos.png
│   ├── comparacion_arquitecturas_DL.png
│   ├── curvas_convergencia_DL.png
│   ├── matriz_confusion.png
│   ├── validacion_vs_prueba.png
│   ├── shap_importancia.png
│   └── shap_detalle.png
└── README.md
```

## DATASET

**Healthcare Stroke Dataset** — Kaggle (fedesoriano, 2021)  
5.110 registros | 11 variables clinicas y demograficas | Desbalance: 95% no stroke / 5% stroke

Variables: edad, hipertension, enfermedad cardiaca, glucosa, BMI, genero, estado civil, tipo de trabajo, tipo de residencia, habito de fumar, stroke (variable objetivo).


## MODELOS EVALUADOS

| MODELO                       | AUC-ROC   | Recall   | F1        |
|------------------------------|-----------|----------|-----------|
| Random Forest                | 0.785     | 0.00     | 0.000     |
| **Regresion Logistica**      | **0.840** | **0.80** | **0.232** |
| XGBoost                      | 0.764     | 0.12     | 0.129     |
| MLP Base (64-32-16-1)        | 0.825     | 0.78     | 0.218     |
| MLP L2 (64-32-16-1)          | 0.800     | 0.76     | 0.212     |
| MLP Profundo(128-64-32-16-1) | 0.832     | 0.82     | 0.240     |

**Modelo seleccionado:** Regresion Logistica con class_weight='balanced'


## TECNICAS DE BALANCEO COMPARADAS

| Tecnica                 | Recall | AUC   |
|-------------------------|--------|-------|
| class_weight='balanced' | 0.80   | 0.840 |
| SMOTE                   | 0.80   | 0.841 |
| Undersampling           | 0.82   | 0.835 |



## INSTALACION Y EJECUCION

### Requisitos
```bash
Python 3.8+
```

### Instalar dependencias
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap tensorflow streamlit joblib matplotlib seaborn
```

### Ejecutar el notebook
1. Abrir `notebooks/acif104_s9_grupo04_Arganaraz_Ferrada_Gonzalez.ipynb` en Google Colab o Jupyter
2. Si usa Google Colab, montar Google Drive y actualizar la ruta del dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/acif104_s9_grupo04/healthcare-dataset-stroke-data.csv')
```
3. Ejecutar todas las celdas en orden

### Ejecutar la aplicacion Streamlit
```bash
cd scripts
streamlit run app.py
```
La aplicacion estara disponible en: http://localhost:8501


## APLICACION WEB

La app Streamlit permite:
- Ingresar datos clinicos del paciente mediante interfaz interactiva
- Obtener prediccion de riesgo de stroke con probabilidad (%)
- Ver las variables mas influyentes segun SHAP

**Modelo backend:** Regresion Logistica serializada (`modelos_entrenados/modelo_stroke.pkl`)  
**Scaler:** StandardScaler serializado (`modelos_entrenados/scaler_stroke.pkl`)

---

## RESULTADOS FASE 3

- Particion formal en 3 conjuntos estratificados (65% / 15% / 20%)
- Analisis de convergencia: 3 arquitecturas DL sin sobreajuste (diff loss < 0.05)
- Validacion cruzada confirma consistencia de Regresion Logistica (diff AUC val/test < 0.02)
- Matriz de confusion: 40/50 casos stroke detectados correctamente (80% recall)

---

## TECNOLOGIAS

Python · scikit-learn · XGBoost · TensorFlow/Keras · imbalanced-learn · SHAP · Streamlit · pandas · numpy · matplotlib · seaborn · joblib

---

## REFERENCIAS

Fedesoriano (2021). Stroke Prediction Dataset. Kaggle.  
Fernandes et al. (2024). Sensors, 24(13), 4355.  
Mainali et al. (2021). Frontiers in Neurology, 12, 734345.  
Geron, A. (2022). Hands-on Machine Learning. O'Reilly Media.  
Goodfellow et al. (2016). Deep Learning. MIT Press.
