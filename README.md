# RiskAlert-Diabetes

# 🔍 Modelo de Predicción de Diabetes con Machine Learning

Este proyecto utiliza inteligencia artificial para predecir el riesgo de diabetes en función de datos de salud y estilo de vida. Con un modelo de Machine Learning y un backend en Flask, permite a los usuarios ingresar información clave y recibir un diagnóstico (No diabético, Pre-diabético o Diabético), acompañado de recomendaciones personalizadas generadas por IA.

## 📚 Descripción

El modelo fue entrenado con un conjunto de datos que incluye información categórica relacionada con el estilo de vida, estado de salud y características personales. Se aplicaron técnicas de preprocesamiento como limpieza de datos, manejo de valores atípicos, codificación de variables categóricas y balanceo de clases para asegurar un entrenamiento justo y representativo.

Se exploraron diferentes modelos de clasificación y se compararon con métricas clave para identificar el de mejor desempeño.

Fuente de los datos: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

# 🚀 Características

✅ Predicción rápida y precisa del riesgo de diabetes.

✅ Interfaz web sencilla con Tailwind CSS y HTML.

✅ Backend en Flask con integración de un modelo de ML.

✅ Conexión con la API de Gemini para generar recomendaciones de salud.

## ⚙️ Modelos utilizados
- Regresión logísitca
- Árboles de Decisión
- Random Forest
- K-Nearest Neighbors (KNN)-
- XGboost
- LightGBM

## 📈 Métricas de evaluación

Se utilizaron las siguientes métricas para evaluar el desempeño de los modelos:

- Accuracy
- Precision (por clase)
- Recall (por clase)
- F1-score (por clase)
- Specificity (por clase)

El mejor modelo fue el Árbol de Decisión usando la base de datos que tiene hasta 3 desviaciones estandar y balanceada, con un **accuracy aproximado del 78%**, mostrando un buen balance entre sensibilidad y especificidad en todas las clases.

## 🛠 Tecnologías utilizadas

- Python (Flask, Scikit-Learn, Pandas)

- HTML + Tailwind CSS (Interfaz web)

- Google Gemini API (Generación de recomendaciones personalizadas)

- GitHub (Control de versiones y despliegue)


## 📌 Cómo ejecutar el proyecto

#### 1️⃣ Instala las dependencias:

pip install -r requirements.txt

#### 2️⃣  Ejecuta la aplicación Flask:

python app.py

#### 3️⃣ Abre el navegador en:

http://127.0.0.1:5000/

## 🚀 Posibles mejoras futuras

- Ampliar el dataset con nuevas variables clínicas y demográficas
- Explorar modelos más avanzados como redes neuronales profundas
- Integrar el modelo en una aplicación web o dashboard interactivo
- Validación en entornos reales (clínicas, hospitales, EPS)

💡 Este proyecto busca ayudar en la detección temprana de la diabetes y fomentar hábitos de vida saludables. ¡Contribuciones y mejoras son bienvenidas!
