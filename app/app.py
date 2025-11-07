# Contenido para: app/app.py

import streamlit as st
import requests
import pandas as pd

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Coach de Bienestar Preventivo",
    page_icon="❤️",
    layout="centered"
)

# --- 1. Definición de la UI (Basado en tu HTML) ---
st.title("Coach de Bienestar Preventivo")
st.write("Ingresa tus datos para estimar tu riesgo y recibir un plan de acción personalizado por nuestra IA Híbrida.")

# URL de la API de FastAPI (que debe estar corriendo en otra terminal)
API_URL = "http://127.0.0.1:8000"

# --- 2. El Formulario de Streamlit ---
with st.form(key='wellness_form'):
    st.header("1. Tus Métricas Básicas")
    
    # Dividir en columnas para un mejor layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Edad (en años)', min_value=18, max_value=120, value=45, help="Tu edad actual.")
        height_cm = st.number_input('Altura (en centímetros)', min_value=100, max_value=250, value=170, help="Ej: 170")
    
    with col2:
        sex = st.selectbox('Sexo Biológico', ['Masculino', 'Femenino'], index=0, help="El sexo biológico es usado para los cálculos del modelo.")
        weight_kg = st.number_input('Peso (en kilogramos)', min_value=30.0, max_value=300.0, value=75.5, step=0.1, help="Ej: 75.5")

    waist_cm = st.number_input('Circunferencia de Cintura (en centímetros)', min_value=50.0, max_value=200.0, value=90.0, help="Mide tu cintura al nivel del ombligo.")

    st.header("2. Tus Hábitos Diarios")
    
    col3, col4 = st.columns(2)
    
    with col3:
        sleep_hours = st.number_input('Horas de sueño promedio', min_value=0.0, max_value=24.0, value=7.0, step=0.5, help="¿Cuántas horas duermes en una noche típica?")
        smokes_cig_day = st.number_input('¿Fumas?', min_value=0, max_value=100, value=0, help="¿Cuántos cigarrillos por día? (0 si no fumas)")
        
    with col4:
        days_mvpa_week = st.number_input('Días de actividad física', min_value=0, max_value=7, value=3, help="Días por semana con al menos 30 min de actividad moderada.")
        # Omitimos 'fruit_veg_portions_day' porque nuestro modelo ML no fue entrenado con él,
        # pero podríamos añadirlo al prompt del Coach RAG si quisiéramos.

    # El botón de envío
    st.divider()
    submit_button = st.form_submit_button(label='Calcular Riesgo y Obtener Plan', use_container_width=True)

# --- 3. Lógica de Backend (Cuando se presiona el botón) ---
if submit_button:
    
    # Validaciones (¡Importante!)
    if height_cm == 0 or weight_kg == 0 or waist_cm == 0:
        st.error("Por favor, ingresa valores válidos para Altura, Peso y Cintura.")
    else:
        with st.spinner('Contactando a la IA... Analizando tu perfil...'):
            try:
                # --- A. Preparar los datos para el Cerebro 1 (ML) ---
                # El formulario (frontend) nos da datos "crudos".
                # Debemos convertirlos a las 7 "features" que el modelo ML espera.
                
                feat_imc = weight_kg / (height_cm / 100) ** 2
                feat_whtr = waist_cm / height_cm
                feat_sex = 0 if sex == 'Masculino' else 1
                feat_is_smoker = 1 if smokes_cig_day > 0 else 0
                
                # El payload para el endpoint /predict
                predict_payload = {
                    "feat_imc": feat_imc,
                    "feat_whtr": feat_whtr,
                    "feat_age": age,
                    "feat_sex": feat_sex,
                    "feat_is_smoker": feat_is_smoker,
                    "feat_sleep_hours": sleep_hours,
                    "feat_activity_days": days_mvpa_week
                }

                # --- B. Llamar al Cerebro 1 (Endpoint /predict) ---
                predict_response = requests.post(f"{API_URL}/predict", json=predict_payload)
                predict_response.raise_for_status() # Lanza un error si la API falla
                
                prediction_data = predict_response.json()
                risk_score = prediction_data.get("risk_score", 0)
                prediction = prediction_data.get("prediction", 0)

                # Mostrar el primer resultado
                st.subheader("Resultado del Análisis de Riesgo (ML)")
                risk_percent = risk_score * 100
                
                if prediction == 1:
                    st.error(f"**Riesgo Detectado: Alto ({risk_percent:.1f}%)**")
                else:
                    st.success(f"**Riesgo Detectado: Bajo ({risk_percent:.1f}%)**")

                # --- C. Llamar al Cerebro 2 (Endpoint /coach) ---
                # Usamos el resultado del Cerebro 1 para alimentar al Cerebro 2
                with st.spinner('Generando tu plan de acción personalizado...'):
                    coach_response = requests.post(f"{API_URL}/coach", json=prediction_data)
                    coach_response.raise_for_status()
                    
                    coach_data = coach_response.json()
                    coach_message = coach_data.get("coach_message", "No se pudo generar un consejo.")

                    st.subheader("Plan de Acción del Coach IA (RAG)")
                    st.markdown(coach_message) # Usamos markdown para un texto bonito
            
            except requests.exceptions.ConnectionError:
                st.error("¡Error de Conexión! No se pudo conectar a la API.")
                st.info("Asegúrate de que el servidor de Uvicorn (api/main.py) esté corriendo en la Terminal 1.")
            except Exception as e:
                st.error(f"Ha ocurrido un error inesperado: {e}")