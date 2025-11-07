import streamlit as st
import pandas as pd
import base64
import os
import datetime
import sys 
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any

# =========================================================================
# === CRÍTICO: FORZAR LECTURA DE LA CLAVE DE ENTORNO (HUGGING FACE) ===
# Establece la variable de entorno OPENAI_API_KEY al inicio para LangChain
# y el resto de librerías puedan leerla desde el sistema operativo.
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
# =========================================================================


# Importaciones de reportlab (para generar el PDF)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black

# --- CONFIGURACIÓN CRÍTICA PARA IMPORTAR src/inference.py ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Importaciones de Librerías y Lógica ---
try:
    from src.inference import load_ml_model, load_rag_system, get_risk_score, generate_rag_response
    from src.prompts import RAG_PROMPT_TEMPLATE 

except ImportError as e:
    st.error(f"ERROR CRÍTICO DE IMPORTACIÓN: No se pudo cargar la lógica de src. Detalles: {e}. Revisa tus archivos 'src/inference.py' y 'src/prompts.py'")
    st.stop()


# --- CONSTANTES Y CONFIGURACIÓN ---

RISK_THRESHOLD_HIGH = 0.65 
MODEL_PATH = "models/hypertension_model.joblib"
# CRÍTICO: COLUMNAS ESPERADAS POR EL MODELO ML
FEATURE_COLS = [
    'Age', 'Gender', 'Weight_kg', 'Height_m', 'Sleep_Hours', 
    'Cigarettes_Day', 'Days_MVPA_Week', 'Fruit_Veg_Portions', 'Waist_cm'
]


st.set_page_config(
    page_title="NexusByte: Coach de Bienestar Preventivo IA Híbrida",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS (Se mantiene)
st.markdown("""
<style>
    .header-text {
        color: #007BFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .chat-container {
        border-radius: 10px;
        background-color: white;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- FUNCIONES DE CARGA Y CACHÉ ---

@st.cache_resource
def get_ml_model():
    """Carga el modelo de Machine Learning y lo cachea para eficiencia."""
    try:
        model = load_ml_model(MODEL_PATH) 
        return model
    except Exception as e:
        print(f"Error al cargar ML Model en app: {e}")
        return None

@st.cache_resource
def get_rag_system():
    """Carga el sistema RAG (retriever y LLM) y lo cachea."""
    try:
        retriever, llm = load_rag_system()
        return retriever, llm
    except Exception as e:
        print(f"Error al cargar RAG System en app: {e}")
        return None, None

# --- GENERACIÓN DE PDF (Se mantiene) ---

def create_pdf_report(user_data: Dict[str, Any], risk_score: float, drivers: List[str], plan_content: str) -> bytes:
    """Genera el documento PDF del plan personalizado."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    Story = []

    styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, spaceAfter=20, alignment=1, textColor=HexColor('#007BFF')))
    styles.add(ParagraphStyle(name='SubTitleStyle', fontSize=14, spaceAfter=15, textColor=HexColor('#4CAF50')))
    styles.add(ParagraphStyle(name='DisclaimerStyle', fontSize=10, spaceBefore=30, textColor=black))
    styles.add(ParagraphStyle(name='NormalStyle', fontSize=12, leading=16))

    Story.append(Paragraph("Reporte de Bienestar Preventivo - NexusByte", styles['TitleStyle']))

    Story.append(Paragraph("1. Perfil y Estimación de Riesgo", styles['SubTitleStyle']))
    Story.append(Paragraph(f"**Fecha del Reporte:** {datetime.date.today().strftime('%d/%m/%Y')}", styles['NormalStyle']))
    Story.append(Paragraph(f"**Score de Riesgo (0-1):** <font color='#007BFF'>{risk_score:.2f}</font>", styles['NormalStyle']))
    Story.append(Paragraph(f"**Factores Clave (Drivers):** {', '.join(drivers)}", styles['NormalStyle']))
    Story.append(Spacer(1, 0.2*inch))

    Story.append(Paragraph("2. Plan de Acción Personalizado (Coach IA)", styles['SubTitleStyle']))
    formatted_plan = plan_content.replace('\n', '<br/>').replace('*', '') 
    Story.append(Paragraph(formatted_plan, styles['NormalStyle']))
    Story.append(Spacer(1, 0.5*inch))

    disclaimer = "⚠️ **DISCLAIMER:** Este reporte es generado por un sistema de Inteligencia Artificial Preventiva y NO constituye un diagnóstico médico. Siempre debe consultar a un profesional de la salud (médico, nutricionista o kinesiólogo) para cualquier decisión o plan de tratamiento."
    Story.append(Paragraph(disclaimer, styles['DisclaimerStyle']))

    doc.build(Story)
    return buffer.getvalue()


# --- FUNCIONES AUXILIARES ---

def get_mock_drivers(user_data: Dict[str, Any]) -> List[str]:
    """Retorna factores clave simulados basados en las entradas del usuario."""
    drivers = []
    if user_data.get('sleep_hours', 7.5) < 6.5:
        drivers.append("Sueño Insuficiente")
    if user_data.get('smokes_cig_day', 0) > 0:
        drivers.append("Tabaquismo")
    if user_data.get('days_mvpa_week', 3) < 3:
        drivers.append("Baja Actividad Física")
    if user_data.get('fruit_veg_portions_day', 5.0) < 5.0:
        drivers.append("Ingesta Baja de Frutas/Verduras")
    if user_data.get('waist_cm', 90.0) > 102.0 and user_data['sex_code'] == 1: 
        drivers.append("Circunferencia de Cintura Elevada")
    elif user_data.get('waist_cm', 90.0) > 88.0 and user_data['sex_code'] == 0: 
        drivers.append("Circunferencia de Cintura Elevada")

    return drivers if drivers else ["Perfil General Saludable"]


# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---

def main():
    st.markdown("<h1 class='header-text'>Coach de Bienestar Preventivo IA Híbrida</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Carga de Modelos
    ml_model = get_ml_model() 
    retriever, llm = get_rag_system() 

    # 2. Sidebar para Configuración/Estado
    st.sidebar.markdown("## ⚙️ Estado del Sistema Híbrido")
    st.sidebar.markdown(f"**Estado del Modelo ML:** {'✅ Listo' if ml_model is not None else '❌ No Cargado'}")
    # CRÍTICO: Si falló, mostramos el mensaje de que la clave NO fue inyectada
    if retriever is None or llm is None:
        st.sidebar.markdown(f"**Estado del Sistema RAG:** ❌ No Cargado (Clave no inyectada en el entorno)")
        st.error("No se pudieron cargar todos los componentes. Revisa tu clave en las Configuraciones (Secrets) de Hugging Face.")
    else:
        st.sidebar.markdown(f"**Estado del Sistema RAG:** ✅ Listo")
        

    # 3. Formulario de Entrada y Score ML
    with st.container():
        st.markdown("## 📝 1. Ingreso de Perfil y Estimación de Riesgo")
        
        user_data = {}
        col1, col2, col3 = st.columns(3)

        with col1:
            user_data['age'] = st.slider("Edad (años)", min_value=18, max_value=85, value=45)
            user_data['sex'] = st.selectbox("Sexo Biológico", options=['Masculino', 'Femenino'], index=0)
            user_data['sex_code'] = 1 if user_data['sex'] == 'Masculino' else 0 
        
        with col2:
            user_data['height_cm'] = st.number_input("Altura (cm)", min_value=120, max_value=220, value=175)
            user_data['weight_kg'] = st.number_input("Peso (kg)", min_value=30.0, max_value=220.0, value=80.0, step=0.5)
            user_data['waist_cm'] = st.number_input("Cintura (cm)", min_value=40.0, max_value=170.0, value=90.0, step=0.5)

        with col3:
            user_data['sleep_hours'] = st.slider("Horas de Sueño/Día", min_value=3.0, max_value=14.0, value=7.5, step=0.1)
            user_data['smokes_cig_day'] = st.number_input("Cigarros/Día", min_value=0, max_value=60, value=0)
            user_data['days_mvpa_week'] = st.slider("Días con Actividad Física Vigorosa/Semana", min_value=0, max_value=7, value=3)
            user_data['fruit_veg_portions_day'] = st.slider("Porciones Frutas/Verduras/Día", min_value=0.0, max_value=12.0, value=5.0, step=0.5)

        
        ml_features = {
            'Age': user_data['age'], 
            'Gender': user_data['sex_code'], 
            'Weight_kg': user_data['weight_kg'],
            'Height_m': user_data['height_cm'] / 100, 
            'Sleep_Hours': user_data['sleep_hours'],
            'Cigarettes_Day': user_data['smokes_cig_day'],
            'Days_MVPA_Week': user_data['days_mvpa_week'],
            'Fruit_Veg_Portions': user_data['fruit_veg_portions_day'],
            'Waist_cm': user_data['waist_cm']
        }
        
        if st.button("📊 Estimar Riesgo Cardiometabólico", type="primary"):
            if ml_model is not None:
                risk_score_value = get_risk_score(ml_model, ml_features, FEATURE_COLS) 

                if risk_score_value >= 0:
                    drivers = get_mock_drivers(user_data) 
                    st.session_state['risk_score'] = risk_score_value
                    st.session_state['drivers'] = drivers
                    st.session_state['user_data'] = user_data
                else:
                    st.error("Error al calcular el riesgo. Asegúrate que tu modelo ML sea compatible con las features ingresadas.")
            else:
                st.error("No se puede calcular el riesgo. El Modelo ML no está cargado correctamente.")
    
    st.markdown("---")
    
    # 4. Resultados ML y Coach IA
    if 'risk_score' in st.session_state:
        
        risk_score = st.session_state['risk_score']
        drivers = st.session_state['drivers']
        user_data = st.session_state['user_data']
        
        st.markdown("## ✨ 2. Resultados y Coach de Bienestar")
        
        col_score, col_coach_chat = st.columns([1, 2])
        
        with col_score:
            
            if risk_score > RISK_THRESHOLD_HIGH:
                message = "⚠️ **RIESGO ALTO:** Probabilidad elevada. **CONSULTAR a un profesional.**"
            elif risk_score > 0.4:
                message = "**RIESGO MODERADO:** Enfoque en mejorar hábitos. Contacte a un especialista."
            else:
                message = "**RIESGO BAJO:** Perfil saludable. ¡Mantenga los buenos hábitos!"

            st.metric(label="Puntaje de Riesgo (0-1)", value=f"{risk_score:.3f}")
            st.markdown(f"<div style='border: 1px solid #ccc; padding: 15px; border-radius: 10px; background-color: white;'>{message}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**Factores Impulsores del Riesgo:**")
            for driver in drivers:
                st.markdown(f"- {driver}")
                
            plan_content = st.session_state.get('plan_content', "Aún no se ha generado un plan personalizado en el chat del Coach.")
            
            if plan_content != "Aún no se ha generado un plan personalizado en el chat del Coach.":
                pdf_bytes = create_pdf_report(user_data, risk_score, drivers, plan_content)
                
                st.download_button(
                    label="📥 Descargar Plan Personalizado (PDF)",
                    data=pdf_bytes,
                    file_name="NexusByte_Plan_Bienestar.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("Pídele al Coach que genere tu plan para habilitar la descarga del PDF.")

        
        with col_coach_chat:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            st.markdown("### 💬 Coach IA: Preguntas y Plan de Acción")
            
            if 'messages' not in st.session_state:
                st.session_state.messages = []
                initial_message = f"Hola! Soy tu Coach de Bienestar NexusByte. Acabas de obtener un riesgo de **{risk_score:.2f}**. Mi objetivo es ayudarte a crear un plan de 2 semanas basado en tus factores clave ({', '.join(drivers)}). ¿Qué pregunta tienes o quieres que **genere tu plan de inmediato**?"
                st.session_state.messages.append({"role": "assistant", "content": initial_message})

            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Pregúntale a tu Coach (ej: 'Quiero mi plan de 2 semanas')"):
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Procesando consulta y buscando en la base de conocimiento..."):
                        
                        llm_query = f"Consulta del usuario: '{prompt}'. Datos del perfil: Edad={user_data['age']}, Sexo={user_data['sex']}, Peso={user_data['weight_kg']}kg, Riesgo={risk_score:.2f}. Factores clave del riesgo: {', '.join(drivers)}."
                        
                        if retriever is not None and llm is not None:
                            response = generate_rag_response(
                                llm, 
                                retriever, 
                                llm_query 
                            )
                        else:
                            # CRÍTICO: Si falla, informamos que no es por el ML sino por la clave
                            response = "Lo siento, el Coach IA no está disponible. Por favor, verifica que la clave de OpenAI API esté configurada correctamente."
                            
                        st.markdown(response)
                        
                        if "plan" in prompt.lower() or "recomendación" in prompt.lower():
                            st.session_state['plan_content'] = response
                            st.info("✅ Plan de acción guardado. Ya puedes descargar el PDF.")
                            
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()