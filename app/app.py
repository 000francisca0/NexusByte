import streamlit as st
import pandas as pd
import base64
import os
import json
import datetime
import time 
import sys # Necesario para añadir la ruta de la lógica

# --- CONFIGURACIÓN CRÍTICA PARA IMPORTAR src/inference.py ---
# Añade el directorio 'src' al path para que el intérprete de Python 
# pueda encontrar el módulo 'src.inference' que contiene la lógica.
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# Las importaciones de src/inference.py ya se encargan de cargar
# los modelos y el RAG una sola vez, mejorando la performance.
try:
    from src.inference import get_risk_score, get_risk_drivers, get_coach_response, ml_model, rag_chain
    # Si los modelos no cargaron (error en joblib o faiss), ml_model o rag_chain serán None
    models_ready = ml_model is not None and rag_chain is not None
except ImportError as e:
    st.error(f"❌ ERROR CRÍTICO DE IMPORTACIÓN: No se pudo cargar src/inference.py. {e}")
    models_ready = False
except Exception as e:
    st.error(f"❌ ERROR CRÍTICO al inicializar la lógica: {e}")
    models_ready = False

# Importaciones de reportlab (para generar el PDF)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black
    from io import BytesIO
    reportlab_ok = True
except ImportError:
    st.warning("⚠️ No se pudo importar ReportLab. La funcionalidad de PDF no estará disponible.")
    reportlab_ok = False

# --- CONSTANTES ---
# La ruta de la imagen es relativa a la raíz del repositorio, no a app.py
IMAGE_PATH = "app/coach_image.png"
TITLE_COLOR = "#0D7377" # Turquesa oscuro
ACCENT_COLOR = "#E0F7FA" # Celeste muy claro

# --- ESTILOS DE STREAMLIT ---
st.set_page_config(
    page_title="NexusByte - Coach de Bienestar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para Streamlit
st.markdown(f"""
<style>
    .stApp {{
        background-color: {ACCENT_COLOR}; 
    }}
    .title-text {{
        color: {TITLE_COLOR};
        font-size: 2.5em;
        font-weight: bold;
    }}
    .stButton>button {{
        background-color: {TITLE_COLOR};
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #0A5F61;
    }}
    .stProgress > div > div > div > div {{
        background-color: {TITLE_COLOR};
    }}
    .css-15zrgz7 {{
        color: {TITLE_COLOR};
    }}
</style>
""", unsafe_allow_html=True)


# --- FUNCIONES CORE ---

def display_explanation_box(risk_score, drivers):
    """Muestra el puntaje de riesgo y los factores clave."""
    if not drivers:
        drivers = ["Datos incompletos o atípicos.", "Se recomienda ingresar todos los valores para un análisis completo."]
    
    st.markdown(f'<div class="title-text">Tu Perfil de Riesgo Cardiometabólico</div>', unsafe_allow_html=True)
    
    # Mapeo de riesgo y color (Guardrail 4: Umbral de derivación)
    if risk_score >= 0.7:
        color = "red"
        risk_level = "¡ALTO RIESGO!"
        message = "🔴 Se detecta un **riesgo alto**. Es fundamental que consultes con un profesional de salud de inmediato. Nuestro coach te dará un plan de prevención."
    elif risk_score >= 0.3:
        color = "orange"
        risk_level = "Riesgo Moderado"
        message = "🟠 Tienes un riesgo moderado. Hay margen de mejora en tus hábitos. Usa el Coach para generar tu plan."
    else:
        color = "green"
        risk_level = "Riesgo Bajo"
        message = "🟢 Tu riesgo es bajo. Mantén tus hábitos saludables. Usa el Coach para optimizar tu bienestar."

    # Muestra el score y el nivel de riesgo
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-top: 20px;">
        <p style="font-size: 1.5em; margin-bottom: 5px;">Puntaje de Riesgo (0-1): <b style="color: {color};">{risk_score:.3f}</b></p>
        <p style="font-size: 1.2em; margin-top: 0;">Nivel: <b>{risk_level}</b></p>
        <p style="font-size: 0.9em;">{message}</p>
        
        <h3 style="color: {TITLE_COLOR}; margin-top: 20px; font-size: 1.1em;">Factores Clave que Impulsan tu Riesgo:</h3>
        <ul style="list-style-type: disc; margin-left: 20px;">
            {''.join([f'<li>{d}</li>' for d in drivers])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Inicializa el chat después del score
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "coach", "content": f"Hola, soy tu Coach de Bienestar. He analizado tu perfil con un riesgo de **{risk_score:.3f}**. ¿Te gustaría un plan de acción personalizado o tienes preguntas sobre cómo mejorar tu salud?"})
    
    st.subheader("💬 Consulta con tu Coach Personalizado (IA Híbrida)")
    
    # Renderiza el historial de chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_pdf(risk_score, drivers, plan_text):
    """Genera un plan de bienestar personalizado en formato PDF."""
    if not reportlab_ok:
        st.error("Error: La librería ReportLab no está disponible para generar el PDF.")
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, spaceAfter=20, alignment=1, textColor=HexColor(TITLE_COLOR)))
    styles.add(ParagraphStyle(name='Heading2', fontName='Helvetica-Bold', fontSize=14, spaceBefore=10, spaceAfter=10, textColor=HexColor(TITLE_COLOR)))
    styles.add(ParagraphStyle(name='BodyText', fontName='Helvetica', fontSize=10, spaceBefore=6))
    styles.add(ParagraphStyle(name='DisclaimerStyle', fontName='Helvetica-Oblique', fontSize=8, textColor=black))

    flowables = []
    
    # Título
    flowables.append(Paragraph("PLAN DE BIENESTAR PREVENTIVO", styles['TitleStyle']))
    
    # Datos de Riesgo
    flowables.append(Paragraph(f"<b>Fecha de Reporte:</b> {datetime.date.today().strftime('%d/%m/%Y')}", styles['BodyText']))
    flowables.append(Paragraph(f"<b>Puntaje de Riesgo Cardiometabólico (0-1):</b> <font color='{TITLE_COLOR}'>{risk_score:.3f}</font>", styles['BodyText']))
    flowables.append(Spacer(0, 12))
    
    # Factores Clave
    flowables.append(Paragraph("FACTORES CLAVE DEL RIESGO", styles['Heading2']))
    flowables.append(Paragraph("Estas son las principales variables que contribuyen a tu puntaje de riesgo:", styles['BodyText']))
    
    list_items = [Paragraph(f"• {d}", styles['BodyText']) for d in drivers]
    flowables.extend(list_items)
    flowables.append(Spacer(0, 12))
    
    # Plan de Acción (El texto generado por el LLM)
    flowables.append(Paragraph("PLAN DE ACCIÓN DE 2 SEMANAS (Recomendaciones del Coach)", styles['Heading2']))
    
    # Procesar el texto plano del LLM en párrafos
    plan_paragraphs = plan_text.split('\n')
    for p in plan_paragraphs:
        if p.strip():
            flowables.append(Paragraph(p.strip(), styles['BodyText']))
    
    flowables.append(Spacer(0, 24))
    
    # Disclaimer (Guardrail 1: No diagnóstico médico)
    disclaimer = """
    <b>DISCLAIMER MÉDICO OBLIGATORIO:</b> Este plan y el puntaje de riesgo son generados por un modelo de Inteligencia Artificial (IA) con fines exclusivamente preventivos y educativos. NO constituyen un diagnóstico médico, NI reemplazan la consulta o la evaluación de un profesional de la salud (médico, nutricionista, kinesiólogo, etc.). Si tienes síntomas o un riesgo alto, consulta a tu médico.
    """
    flowables.append(Paragraph(disclaimer, styles['DisclaimerStyle']))
    
    doc.build(flowables)
    buffer.seek(0)
    return buffer


def handle_coach_chat(user_input):
    """Maneja la interacción del chatbot y actualiza el estado."""
    if not models_ready:
        st.session_state.chat_history.append({"role": "coach", "content": "Lo siento, el modelo ML o el sistema RAG no se cargaron correctamente. Por favor, contacta al equipo de soporte."})
        return # <-- Aquí termina la función, evitando el error de sintaxis

    # Añade el mensaje del usuario al historial
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Llama al endpoint del coach (usando la función de inference.py)
    with st.spinner("🤖 El Coach de Bienestar está pensando..."):
        # Asegúrate de pasar el historial para que el LLM tenga contexto
        full_query = f"Historial: {st.session_state.chat_history[-2].get('content', 'N/A')}\nUsuario: {user_input}"
        
        # Aquí la llamada debe ser al LLM con RAG
        coach_response_text = get_coach_response(full_query)

    # Añade la respuesta del coach al historial
    st.session_state.chat_history.append({"role": "coach", "content": coach_response_text})
    
    # Esto es vital para forzar la re-renderización del chat
    st.rerun()


def handle_pdf_generation(risk_score, drivers):
    """Genera el plan de acción (llamando al LLM) y habilita la descarga."""
    if not models_ready:
        st.error("Error: Los modelos no están cargados. No se puede generar el plan.")
        return # <-- Aquí termina la función, evitando el error de sintaxis

    st.session_state.generating_plan = True
    
    with st.spinner("🤖 Generando Plan de Acción y Citas..."):
        # El LLM genera el plan de 2 semanas con RAG
        # Creamos un prompt específico para el plan
        plan_prompt = (
            f"Basado en el perfil de riesgo {risk_score:.3f} y los factores clave ({', '.join(drivers)}), "
            "genera un plan de acción SMART de 2 semanas. El plan debe incluir recomendaciones de "
            "Sueño, Actividad Física y Dieta, citando exclusivamente la base de conocimiento local (RAG). "
            "Formatea la respuesta como texto plano con saltos de línea y encabezados."
        )
        
        # Obtenemos la respuesta del LLM con RAG
        plan_text = get_coach_response(plan_prompt)

    st.session_state.plan_text = plan_text
    st.session_state.generating_plan = False

    # Generar el PDF final
    pdf_buffer = generate_pdf(risk_score, drivers, plan_text)

    if pdf_buffer:
        st.session_state.pdf_buffer = pdf_buffer
        st.success("✅ Plan de Bienestar generado con éxito. ¡Descárgalo a continuación!")
    else:
        st.error("❌ No se pudo generar el archivo PDF.")


def main():
    """Función principal de Streamlit."""
    
    st.markdown('<div style="text-align: center;"><p class="title-text">Coach de Bienestar Preventivo IA Híbrida</p></div>', unsafe_allow_html=True)

    # --- Sidebar: Imagen y Metadata ---
    with st.sidebar:
        st.image(IMAGE_PATH, caption="NexusByte Health Coach")
        st.markdown("---")
        st.info("Desarrollado para el 1° Hackathon de IA Aplicada Duoc UC 2025 (NHANES).")
        st.markdown(f"**Estado del Modelo ML:** {'✅ Listo' if ml_model else '❌ No Cargado'}")
        st.markdown(f"**Estado del RAG/LLM:** {'✅ Listo' if rag_chain else '❌ No Cargado'}")
        
        # Guardrail: Aviso si los modelos no están listos
        if not models_ready:
            st.error("Modelos no operativos. Verifique logs y requirements.txt.")
            return # <-- Aquí termina la función, evitando el error de sintaxis

    
    # --- Columna 1: Formulario de Entrada ---
    with st.form("input_form"):
        st.subheader("1. Ingresa tu Perfil de Salud")
        
        # Campos de entrada (ajustados al JSON Schema del Hackathon)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad (años)", min_value=18, max_value=85, value=35, step=1)
            height_cm = st.number_input("Altura (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            waist_cm = st.number_input("Circunferencia de Cintura (cm)", min_value=40.0, max_value=170.0, value=85.0, step=0.1)
            sleep_hours = st.number_input("Horas de Sueño/día", min_value=3.0, max_value=14.0, value=7.0, step=0.5)
        
        with col2:
            sex = st.selectbox("Sexo Biológico", ["M", "F"], format_func=lambda x: "Masculino" if x == "M" else "Femenino")
            weight_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=220.0, value=70.0, step=0.1)
            smokes_cig_day = st.number_input("Cigarrillos/día", min_value=0, max_value=60, value=0, step=1)
            days_mvpa_week = st.number_input("Días de Actividad Moderada-Vigorosa / semana", min_value=0, max_value=7, value=3, step=1)
            fruit_veg_portions_day = st.number_input("Porciones de Fruta/Verdura / día", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
            
        
        submitted = st.form_submit_button("🩺 Estimar Riesgo y Activar Coach")

    if submitted:
        # 1. Crear el diccionario de entrada para el modelo ML
        user_data = {
            'age': age,
            'sex': sex,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'waist_cm': waist_cm,
            'sleep_hours': sleep_hours,
            'smokes_cig_day': smokes_cig_day,
            'days_mvpa_week': days_mvpa_week,
            'fruit_veg_portions_day': fruit_veg_portions_day
        }
        
        # 2. Llamar a las funciones de inferencia
        with st.spinner("Calculando riesgo cardiometabólico..."):
            risk_score = get_risk_score(user_data)
            drivers = get_risk_drivers(user_data)
        
        # 3. Guardar en el estado para usar en el chat/PDF
        st.session_state.risk_score = risk_score
        st.session_state.risk_drivers = drivers
        st.session_state.show_results = True
        st.session_state.chat_history = [] # Reinicia el chat al calcular un nuevo riesgo
        st.session_state.pdf_buffer = None # Reinicia el buffer PDF
        st.session_state.plan_text = None
        
        # Forzar la re-ejecución de la app para mostrar los resultados
        st.rerun()

    
    # --- Columna 2: Resultados, Chat y PDF ---
    if st.session_state.get("show_results", False):
        risk_score = st.session_state.risk_score
        drivers = st.session_state.risk_drivers

        # 1. Mostrar explicación del riesgo y factores clave
        display_explanation_box(risk_score, drivers)

        # 2. Sección de Chat
        with st.container():
            user_input = st.chat_input("Escribe tu pregunta o pide un plan de acción...")
            if user_input:
                handle_coach_chat(user_input)

        # 3. Sección de PDF (Exportable)
        st.subheader("4. Exportar Plan de Bienestar Personalizado")
        
        col_pdf_1, col_pdf_2 = st.columns([1, 2])
        
        with col_pdf_1:
            # Botón para generar el plan de acción (llama al LLM)
            if st.button("Generar Plan PDF", disabled=st.session_state.get("generating_plan", False)):
                handle_pdf_generation(risk_score, drivers)

        with col_pdf_2:
            # Botón de descarga (solo aparece si el PDF está listo)
            pdf_buffer = st.session_state.get("pdf_buffer")
            if pdf_buffer:
                st.download_button(
                    label="⬇️ Descargar Plan (PDF)",
                    data=pdf_buffer,
                    file_name="Plan_Bienestar_NexusByte.pdf",
                    mime="application/pdf"
                )
            else:
                st.markdown("Pulsa 'Generar Plan PDF' para crear tu documento.")


if __name__ == "__main__":
    main()