from flask import Flask, render_template, request, jsonify
import numpy as np
import requests
import pandas as pd
import logging
import joblib
import google.generativeai as genai


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize the Flask app
app = Flask(__name__)

ruta_modelo = 'modelo_entrenado.pkl'
ruta_columnas = 'columnas_modelo.pkl'

modelo = joblib.load(ruta_modelo)
columnas = joblib.load(ruta_columnas)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Extraer valores del formulario
    try:
        datos = {
            "edad": int(request.json.get("edad", 0)),
            "presion_arterial_alta": int(request.json.get("presion_arterial_alta", 0)),
            "colesterol_alto": int(request.json.get("colesterol_alto", 0)),
            "imc": float(request.json.get("imc", 0.0)),
            "enfermedad_cardiaca": int(request.json.get("enfermedad_cardiaca", 0)),
            "hace_ejercicio": int(request.json.get("hace_ejercicio", 0)),
            "come_fruta": int(request.json.get("come_fruta", 0)),
            "come_vegetales": int(request.json.get("come_vegetales", 0)),
            "dificultad_caminar": int(request.json.get("dificultad_caminar", 0)),
            "genero": request.json.get("genero", 0),
            "fuma": int(request.json.get("fuma", 0))
        }
    except ValueError:
        return "Error: Datos inválidos en el formulario", 400
    
    # Crear un DataFrame de entrada con los datos del formulario
    entrada = pd.DataFrame([[
        datos["presion_arterial_alta"],
        datos["colesterol_alto"],
        datos["imc"],
        datos["enfermedad_cardiaca"],
        datos["hace_ejercicio"],
        datos["come_fruta"],
        datos["come_vegetales"],
        datos["edad"],
        datos["dificultad_caminar"],
        datos["genero"],
        datos["fuma"]
    ]], columns=columnas)

    # Hacer la predicción con el modelo ML
    resultado = modelo.predict(entrada)[0]
    
    # Asignar etiquetas según el valor de la predicción
    if resultado == 0.0:
        prediccion = "Posible No Diabético"
    elif resultado == 1.0:
        prediccion = "Posible Pre-Diabético"
    else:
        prediccion = "Posible Diabético"

    # Obtener en rango de edad del paciente de la variable categórica para enviar en el prompt
    edades = [
    "Age 18 to 24",
    "Age 25 to 29",
    "Age 30 to 34",
    "Age 35 to 39",
    "Age 40 to 44",
    "Age 45 to 49",
    "Age 50 to 54",
    "Age 55 to 59",
    "Age 60 to 64",
    "Age 65 to 69",
    "Age 70 to 74",
    "Age 75 to 79",
    "Age 80 or older"
    ]
    edad_prompt = edades[datos["edad"]-1]

    # Configurar la API Key
    GEMINI_API_KEY = "Pon la API Key aquí"
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Prompt para pedir recomendación a API de Gemini
    prompt = f"""
    Un paciente de {edad_prompt} años, con:
    - Presión arterial alta: {datos['presion_arterial_alta']}
    - Colesterol alto: {datos['colesterol_alto']}
    - IMC: {datos['imc']}
    - Enfermedad cardíaca: {datos['enfermedad_cardiaca']}
    - Hace ejercicio: {datos['hace_ejercicio']}
    - Come fruta: {datos['come_fruta']}
    - Come vegetales: {datos['come_vegetales']}
    - Tiene dificultad para caminar: {datos['dificultad_caminar']}
    - Género: {datos['genero']}
    - Fuma: {datos['fuma']}
    
    Diagnóstico: {prediccion}.

    Basado en estos datos, proporciona una recomendación de salud para mejorar su bienestar y prevenir complicaciones relacionadas con la diabetes.
    **Da una recomendación breve (máximo 5 líneas) y clara en texto plano**
    """

    # Configurar la API de Gemini
    try:
        # Llamada a la API de Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # Extraer la respuesta generada
        recomendacion = response.text if response.text else "No se pudo obtener la recomendación."
    
    except Exception as e:
        recomendacion = f"Error en la API: {str(e)}"
    return jsonify({"prediccion": prediccion, "recomendacion": recomendacion})

if __name__ == "__main__":
    app.run(debug=True)

