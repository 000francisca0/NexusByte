# Contenido para: app/app.py (Versión 5.6 - Hero Animado Completo)
# Autor: Yanina (NexusByte)
# Desafío Salud NHANES 2025 - Duoc UC
# Mejora: Título verde + imagen circular animada (tipo gif) + layout corregido

import streamlit as st
import requests
import pandas as pd
import base64
import os
import json
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black

# --- 1. Configuración de Página ---
st.set_page_config(
    page_title="Coach de Bienestar NexusByte",
    page_icon="❤️‍🩹",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
IMAGE_PATH = "app/coach_image.png"
LOG_FILE_RESULTS = "app/results.csv"
LOG_FILE_CHAT = "app/logs.jsonl"

# --- 2. CSS Personalizado (Hero animado + colores NexusByte) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
:root {
  --accent:#00BFA6;
  --accent-2:#7AE0C9;
  --muted:#475569;
  --text-dark:#0f172a;
  --bg-1:#f6fffb;
  --bg-2:#eefaf6;
}
html, body {
  font-family:'Inter',system-ui;
  background:linear-gradient(180deg,var(--bg-1),var(--bg-2));
  color:var(--text-dark);
}
.main-container{
  max-width:980px;
  margin:28px auto;
  padding:32px 40px;
  background:linear-gradient(180deg,rgba(255,255,255,0.9),rgba(255,255,255,0.82));
  border-radius:16px;
  box-shadow:0 18px 50px rgba(7,15,25,0.06);
}

/* --- Hero --- */
.hero-left{
  height:250px;
  display:flex;
  flex-direction:column;
  justify-content:center;
}
.hero-left h1{
  color:#00BFA6;
  font-size:2.5rem;
  font-weight:800;
  margin:0 0 10px 0;
  text-shadow:0 2px 6px rgba(0,191,166,0.2);
}
.hero-left p.lead{
  margin:0;
  color:var(--muted);
  font-size:1.2rem;
  line-height:1.6;
}

/* Imagen circular animada */
.avatar-wrap{
  width:250px;
  height:250px;
  border-radius:50%;
  overflow:hidden;
  display:flex;
  align-items:center;
  justify-content:center;
  border:5px solid rgba(255,255,255,0.9);
  background:radial-gradient(circle at 30% 20%,rgba(0,191,166,0.15),transparent 60%);
  box-shadow:0 16px 40px rgba(0,191,166,0.15),0 8px 20px rgba(12,20,30,0.05);
  margin:auto;
  animation:floaty 3s ease-in-out infinite;
}
.avatar-wrap img{
  width:230px;
  height:230px;
  border-radius:50%;
  object-fit:cover;
  transition:transform .4s ease;
}
.avatar-wrap:hover img{
  transform:scale(1.05) rotate(-2deg);
}
@keyframes floaty{
  0%,100%{transform:translateY(-10px);}
  50%{transform:translateY(-22px);}
}

/* --- Chat, botones y expanders --- */
[data-testid="chat-message-container"]{
  border-radius:14px;
  padding:14px 18px;
  margin-bottom:12px;
  box-shadow:0 6px 20px rgba(10,20,30,0.04);
  max-width:86%;
}
[data-testid="chat-message-container"]:not([role="user"]){
  background:linear-gradient(180deg,rgba(0,191,166,0.06),rgba(122,224,201,0.03));
  color:#00332f;
}
[data-testid="chat-message-container"][role="user"]{
  background:#fff;
  color:var(--muted);
  margin-left:auto;
}
.stButton > button, [data-testid="stFormSubmitButton"]{
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  color:#0f172a; 
  font-weight:700;
  border:none;
  border-radius:10px;
  padding:10px 14px;
  transition:all .2s ease;
}
.stButton > button:hover,[data-testid="stFormSubmitButton"]:hover{
  filter:brightness(1.1);
  transform:scale(1.02);
}
[data-testid="stExpander"]{
  border-radius:10px;
  border:1px solid rgba(0,0,0,0.05);
  background:#fcfcfc;
}
#MainMenu,header,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. Funciones de ayuda ---
def log_results(data, prediction_data):
    try:
        file_exists = os.path.isfile(LOG_FILE_RESULTS)
        df_new = pd.DataFrame([data])
        df_new['risk_score'] = prediction_data.get('risk_score', 0)
        df_new['prediction'] = prediction_data.get('prediction', 0)
        df_new['timestamp'] = datetime.datetime.now().isoformat()
        df_new.to_csv(LOG_FILE_RESULTS, index=False, mode='a', header=not file_exists)
    except Exception as e:
        print(f"Error al guardar resultados: {e}")

def log_interaction(query, response):
    try:
        entry = {"timestamp": datetime.datetime.now().isoformat(), "prompt": query, "response": response}
        with open(LOG_FILE_CHAT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error en log del chat: {e}")

def create_pdf(form_data, prediction_data, coach_plan, img_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title', fontName='Helvetica-Bold', fontSize=18, spaceAfter=14, textColor=HexColor("#00BFA6")))
    styles.add(ParagraphStyle(name='Heading2', fontName='Helvetica-Bold', fontSize=14, spaceAfter=10))
    styles.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=12))
    styles.add(ParagraphStyle(name='Disclaimer', fontName='Helvetica-Oblique', fontSize=9, textColor=HexColor("#475569")))
    story = []
    if os.path.exists(img_path):
        story.append(Image(img_path, width=1*inch, height=1*inch))
    story.append(Paragraph("Tu Reporte de Bienestar NexusByte", styles['Title']))
    story.append(Spacer(1, 0.25*inch))
    risk_score = prediction_data.get("risk_score", 0)
    prediction = prediction_data.get("prediction", 0)
    texto = f"Tu riesgo estimado es: {risk_score*100:.1f}%. "
    texto += "(Clasificado como Alto Riesgo)" if prediction==1 else "(Clasificado como Bajo Riesgo)"
    story.append(Paragraph(texto, styles['Body']))
    story.append(Paragraph("Plan de Acción Inicial del Coach", styles['Heading2']))
    story.append(Paragraph(coach_plan.replace("### ","").replace("**","").replace("\n","<br/>"), styles['Body']))
    story.append(Spacer(1,0.25*inch))
    story.append(Paragraph("**IMPORTANTE:** Este reporte es generado por una IA y no constituye un diagnóstico médico.", styles['Disclaimer']))
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph("Datos ingresados:", styles['Heading2']))
    datos = "<ul>" + "".join([f"<li><b>{k.replace('feat_','')}:</b> {v}</li>" for k,v in form_data.items()]) + "</ul>"
    story.append(Paragraph(datos, styles['Body']))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# --- 4. Interfaz Principal ---
image_data_uri = None
if os.path.exists(IMAGE_PATH):
    with open(IMAGE_PATH, "rb") as f:
        image_data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Hero Section ---
col_text, col_image = st.columns([1.5, 1], gap="large")
with col_text:
    st.markdown("""
    <div class="hero-left">
        <h1>Tu Coach de Bienestar Híbrido</h1>
        <p class="lead">Hola — soy <strong>NexusByte</strong>. Te acompaño con consejos, planes y análisis de riesgo. ¿En qué quieres avanzar hoy?</p>
    </div>
    """, unsafe_allow_html=True)
with col_image:
    if image_data_uri:
        st.markdown(f'<div class="avatar-wrap"><img src="{image_data_uri}" alt="NexusByte Avatar"></div>', unsafe_allow_html=True)
    else:
        st.error("No se encontró la imagen del coach.")

# --- 5. Chatbot Interactivo ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content":"¡Hola! Soy NexusByte. Pregúntame sobre hábitos saludables o usa el calculador de riesgo."
    }]
for msg in st.session_state.messages:
    avatar = IMAGE_PATH if msg["role"]=="assistant" and os.path.exists(IMAGE_PATH) else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if 'query_to_process' not in st.session_state:
    st.session_state.query_to_process = None
if len(st.session_state.messages)<=1:
    st.markdown("<h3 style='text-align:center;'>Ideas para comenzar</h3>", unsafe_allow_html=True)
    ideas=["¿Cómo reducir el estrés?","Rutina de ejercicio para principiantes.","Consejos para dormir mejor.","Plan para bajar presión arterial."]
    cols=st.columns(2)
    for i,q in enumerate(ideas):
        if cols[i%2].button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state.query_to_process=q
            st.rerun()

if prompt:=st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.query_to_process=prompt

if st.session_state.query_to_process:
    q=st.session_state.query_to_process
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("assistant", avatar=IMAGE_PATH if os.path.exists(IMAGE_PATH) else None):
        with st.spinner("El coach está pensando..."):
            try:
                payload={"query":q,"history":st.session_state.messages[:-1]}
                resp=requests.post(f"{API_URL}/chat",json=payload,timeout=30)
                resp.raise_for_status()
                msg=resp.json().get("coach_message","No pude procesar la pregunta.")
                st.markdown(msg)
                st.session_state.messages.append({"role":"assistant","content":msg})
                log_interaction(q,msg)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.pop()
    st.session_state.query_to_process=None
    st.rerun()

# --- 6. Calculador de Riesgo ---
st.divider()
with st.expander("📊 Calcular Riesgo de Hipertensión (ML)"):
    st.write("Ingresa tus datos para obtener un análisis de riesgo y un consejo inicial.")
    with st.form("form_risk"):
        c1,c2=st.columns(2)
        with c1:
            age=st.number_input("Edad (años)",18,120,value=None,placeholder="Ej: 45")
            height_cm=st.number_input("Altura (cm)",100,250,value=None,placeholder="Ej: 170")
        with c2:
            sex=st.selectbox("Sexo biológico",["Masculino","Femenino"],index=None,placeholder="Selecciona...")
            weight_kg=st.number_input("Peso (kg)",30.0,300.0,value=None,placeholder="Ej: 70.5")
        waist_cm=st.number_input("Cintura (cm)",50.0,200.0,value=None,placeholder="Ej: 90.0")
        st.caption("Hábitos opcionales:")
        c3,c4=st.columns(2)
        with c3:
            sleep=st.number_input("Horas de sueño",0.0,24.0,value=None,placeholder="Ej: 7.0")
            smoke=st.number_input("¿Fumas? (cig/día)",0,100,value=None,placeholder="Ej: 0")
        with c4:
            activity=st.number_input("Días de actividad física (semana)",0,7,value=None,placeholder="Ej: 3")
        submit=st.form_submit_button("Calcular Riesgo y Obtener Consejo Inicial", use_container_width=True)
    if submit:
        if any(f is None for f in [age,height_cm,sex,weight_kg,waist_cm]):
            st.error("Completa Edad, Altura, Sexo, Peso y Cintura.")
        else:
            with st.spinner("Analizando perfil..."):
                try:
                    imc=weight_kg/(height_cm/100)**2
                    whtr=waist_cm/height_cm
                    sex_val=0 if sex=="Masculino" else 1
                    smoker=1 if smoke and smoke>0 else 0
                    sleep_val=sleep if sleep else 7.0
                    act_val=activity if activity else 3.0
                    payload={
                        "feat_imc":imc,"feat_whtr":whtr,"feat_age":age,
                        "feat_sex":sex_val,"feat_is_smoker":smoker,
                        "feat_sleep_hours":sleep_val,"feat_activity_days":act_val
                    }
                    pr=requests.post(f"{API_URL}/predict",json=payload,timeout=30)
                    pr.raise_for_status()
                    pdata=pr.json()
                    risk_score=pdata.get("risk_score",0)
                    if pdata.get("prediction",0)==1:
                        st.error(f"**Riesgo Alto ({risk_score*100:.1f}%)**")
                        st.warning("Consulta a un profesional de la salud.")
                    else:
                        st.success(f"**Riesgo Bajo ({risk_score*100:.1f}%)**")
                    coach=requests.post(f"{API_URL}/coach",json=pdata,timeout=30)
                    coach.raise_for_status()
                    plan=coach.json().get("coach_message","")
                    st.markdown(plan)
                    st.info("Puedes hacer preguntas sobre este plan en el chat.")
                    log_results(payload,pdata)
                    log_interaction(str(payload),plan)
                    st.session_state.last_result={
                        "form_data":payload,"prediction_data":pdata,"coach_plan":plan
                    }
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 7. Descarga de PDF ---
if "last_result" in st.session_state:
    st.divider()
    st.subheader("Tu Reporte Personalizado")
    res=st.session_state.last_result
    try:
        pdf=create_pdf(res["form_data"],res["prediction_data"],res["coach_plan"],IMAGE_PATH)
        st.download_button(
            "📄 Descargar Plan en PDF",
            data=pdf,
            file_name=f"Reporte_NexusByte_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"No se pudo generar el PDF: {e}")

st.markdown('</div>', unsafe_allow_html=True)
