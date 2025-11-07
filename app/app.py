# Contenido para: app/app.py (Versión 4.1.1 - Arreglo del "key")

import streamlit as st
import requests
import pandas as pd
import base64
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Coach de Bienestar Híbrido",
    page_icon="❤️‍🩹",
    layout="wide"
)

# URL de la API de FastAPI
API_URL = "http://127.0.0.1:8000" 

# --- ¡NUEVO! CÓDIGO DE ESTILO (CSS) ---
st.markdown("""
<style>
    /* Definir nuestro color de marca "lindo" */
    :root {
        --brand-color: #00BFA6; /* Un verde-azulado "saludable" */
        --brand-light: #e0f2f1;
        --text-color: #333;
        --bg-light-gray: #f0f2f5;
        --bg-white: #ffffff;
        --border-color: #e0e0e0;
    }

    /* 1. EL FONDO DE LA PÁGINA */
    body {
        background-color: var(--bg-light-gray);
    }

    /* 2. LA TARJETA PRINCIPAL DE LA APP (El "Marco") */
    .main-container {
        max-width: 850px;
        margin: 40px auto; /* Centra la tarjeta */
        padding: 40px;
        background-color: var(--bg-white);
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    }

    /* 3. LAYOUT DEL ENCABEZADO (Texto | Robot) */
    /* Usaremos esto para centrar verticalmente el texto del saludo */
    #intro-text-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 220px; /* Misma altura que el robot */
    }
    #intro-text {
        font-size: 1.2em;
        line-height: 1.6;
        color: #555;
    }

    /* 4. IMAGEN DEL ROBOT (Grande y Linda) */
    [data-testid="stImage"] > img {
        width: 220px;
        height: 220px;
        border-radius: 50%; /* ¡Círculo! */
        object-fit: cover;
        animation: wave-animation 1.5s ease-in-out;
        display: block;
        margin: auto; /* Centra la imagen en su columna */
    }

    /* 5. FUENTE GENERAL Y TÍTULOS */
    html, body, [class*="st-"] {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
        font-size: 1.2rem;
        color: var(--text-color);
    }
    h1 {
        text-align: center;
        margin-bottom: 30px;
        color: var(--brand-color);
        font-weight: 700;
    }
    h3 { /* "Ideas para comenzar:" */
        color: var(--brand-color);
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* 6. BURBUJAS DE CHAT (LINDAS) */
    [data-testid="chat-message-container"] {
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        max-width: 90%;
        text-align: left !important;
    }
    [data-testid="chat-message-container"]:not([role="user"]) {
        background-color: var(--brand-light);
        color: #004d40;
    }
    [data-testid="chat-message-container"][role="user"] {
        background-color: #f0f0f0;
        color: var(--text-color);
        margin-left: auto;
    }

    /* 7. BOTONES (Ideas para comenzar) */
    .stButton > button {
        background-color: var(--bg-white);
        color: var(--brand-color);
        border: 2px solid var(--brand-color);
        border-radius: 10px;
        padding: 10px 0;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: var(--brand-light);
        color: #004d40;
        border-color: var(--brand-color);
        transform: scale(1.02);
    }

    /* 8. EXPANDER (Arreglo del texto "feo") */
    [data-testid="stExpander"] {
        background-color: #f9f9f9;
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }

    /* Animaciones */
    @keyframes wave-animation {
        0% { transform: rotate(0deg) scale(1); }
        25% { transform: rotate(15deg) scale(1.05); }
        50% { transform: rotate(-10deg) scale(1.05); }
        75% { transform: rotate(5deg) scale(1.05); }
        100% { transform: rotate(0deg) scale(1); }
    }
    
    /* Ocultar pie de página */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Contenedor Principal (La Tarjeta de la App) ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Título de la Aplicación ---
st.markdown("<h1>Tu Coach de Bienestar Híbrido</h1>", unsafe_allow_html=True)

image_path = "app/coach_image.png" 

# --- LAYOUT DE ENCABEZADO (Texto | Robot) ---
col_text, col_robot = st.columns([2, 1], gap="large") 

with col_text:
    st.markdown("""
    <div id="intro-text-container">
        <div id="intro-text">
            <p>¡Hola! Soy <strong>NexusByte</strong>, tu Coach de Bienestar Híbrido.</p>
            <p>Estoy aquí para ayudarte a entender y mejorar tu salud. ¿Cuál es tu consulta el día de hoy?</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_robot:
    if not os.path.exists(image_path):
        st.error("¡No encuentro la imagen en 'app/coach_image.png'!")
    else:
        st.image(image_path, width=220, caption="NexusByte, tu coach IA")


# --- 2. Chatbot Interactivo ---
st.divider() 

if "messages" not in st.session_state:
    st.session_state.messages = [] 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Lógica de Sugerencias (Desaparecen) ---
if 'query_to_process' not in st.session_state:
    st.session_state.query_to_process = None

if not st.session_state.messages:
    st.subheader("Ideas para comenzar:")
    suggestion_cols = st.columns(2) 
    
    suggested_queries = [
        "¿Qué es la dieta DASH?",
        "¿Cómo puedo reducir el estrés?",
        "Rutina de ejercicio para principiantes.",
        "Alimentos buenos para el corazón.",
        "Importancia de dormir bien.",
        "¿Cómo mantenerme motivado?"
    ]

    for i, query in enumerate(suggested_queries):
        if suggestion_cols[i % 2].button(query, key=f"sugg_{i}", use_container_width=True):
            st.session_state.query_to_process = query
            st.rerun() 
    
    st.markdown("<h4 style='text-align: center; margin: 20px 0;'>... o ...</h4>", unsafe_allow_html=True)

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.query_to_process = prompt

# --- Lógica de procesamiento de chat (¡CON MEMORIA!) ---
if st.session_state.query_to_process:
    query = st.session_state.query_to_process
    
    with st.chat_message("user"):
        st.markdown(query)
    
    history_to_send = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("assistant"):
        with st.spinner("El coach está pensando..."):
            try:
                chat_payload = {
                    "query": query,
                    "history": history_to_send 
                }
                
                response = requests.post(f"{API_URL}/chat", json=chat_payload)
                response.raise_for_status()
                
                coach_response = response.json().get("coach_message", "Lo siento, no pude procesar esa pregunta.")
                st.markdown(coach_response)
                
                st.session_state.messages.append({"role": "assistant", "content": coach_response})

            except requests.exceptions.ConnectionError:
                st.error(f"¡Error! No se pudo conectar a la API en {API_URL}.")
                st.info("Asegúrate de que el servidor (uvicorn api.main:app) esté corriendo en la Terminal 1.")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
    
    st.session_state.query_to_process = None
    st.rerun()


# --- SECCIÓN 3: Calculador de Riesgo (Expandible) ---
st.divider()
with st.expander("📊 Calcular Riesgo de Hipertensión (ML)"):
    st.write("Si lo deseas, puedes ingresar tus datos para un análisis de riesgo de Machine Learning.")
    with st.form(key='wellness_form'):
        col1_exp, col2_exp = st.columns(2)
        
        # --- ¡ARREGLO! Se eliminaron los 'key=' de los widgets de aquí abajo ---
        with col1_exp:
            age = st.number_input('Edad (en años)', min_value=18, max_value=120, value=45)
            height_cm = st.number_input('Altura (en cm)', min_value=100, max_value=250, value=170)
        with col2_exp:
            sex = st.selectbox('Sexo Biológico', ['Masculino', 'Femenino'], index=0)
            weight_kg = st.number_input('Peso (en kg)', min_value=30.0, max_value=300.0, value=75.5, step=0.1)

        waist_cm = st.number_input('Cintura (en cm)', min_value=50.0, max_value=200.0, value=90.0)
        
        st.caption("Hábitos (Opcional pero recomendado)")
        col3_exp, col4_exp = st.columns(2)
        with col3_exp:
            sleep_hours = st.number_input('Horas de sueño', min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            smokes_cig_day = st.number_input('¿Fumas?', min_value=0, max_value=100, value=0)
        with col4_exp:
            days_mvpa_week = st.number_input('Días de actividad física', min_value=0, max_value=7, value=3)

        submit_risk_button = st.form_submit_button(label='Calcular Riesgo y Obtener Consejo Inicial', use_container_width=True, key="submit_risk")

    if submit_risk_button:
        # (El resto de la lógica no cambia, ya que lee las variables 'age', 'height_cm', etc.)
        if height_cm == 0 or weight_kg == 0 or waist_cm == 0:
            st.error("Por favor, ingresa valores válidos para Altura, Peso y Cintura.")
        else:
            with st.spinner('Analizando tu perfil...'):
                try:
                    feat_imc = weight_kg / (height_cm / 100) ** 2
                    feat_whtr = waist_cm / height_cm
                    feat_sex = 0 if sex == 'Masculino' else 1
                    feat_is_smoker = 1 if smokes_cig_day > 0 else 0
                    
                    predict_payload = {
                        "feat_imc": feat_imc, "feat_whtr": feat_whtr, "feat_age": age,
                        "feat_sex": feat_sex, "feat_is_smoker": feat_is_smoker,
                        "feat_sleep_hours": sleep_hours, "feat_activity_days": days_mvpa_week
                    }

                    predict_response = requests.post(f"{API_URL}/predict", json=predict_payload)
                    predict_response.raise_for_status()
                    prediction_data = predict_response.json()
                    risk_score = prediction_data.get("risk_score", 0)
                    
                    if prediction_data.get("prediction", 0) == 1:
                        st.error(f"**Riesgo Detectado: Alto ({risk_score*100:.1f}%)**")
                    else:
                        st.success(f"**Riesgo Detectado: Bajo ({risk_score*100:.1f}%)**")

                    with st.spinner('Generando tu plan de acción inicial...'):
                        coach_response = requests.post(f"{API_URL}/coach", json=prediction_data)
                        coach_response.raise_for_status()
                        coach_message = coach_response.json().get("coach_message")
                        
                        st.markdown(coach_message)
                        st.info("¡Puedes chatear con el coach sobre este consejo en la sección de arriba!")

                except requests.exceptions.ConnectionError:
                    st.error(f"¡Error! No se pudo conectar a la API en {API_URL}.")
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado: {e}")

# --- Cerrar Contenedor Principal ---
st.markdown('</div>', unsafe_allow_html=True)