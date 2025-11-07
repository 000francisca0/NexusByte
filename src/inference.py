# Contenido para: src/inference.py
# Propósito: Contiene la lógica del Motor ML y RAG para ser llamada directamente por Streamlit (sin FastAPI).

import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from operator import itemgetter 

# Importaciones de Langchain (copiadas de api/main.py)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuración y Carga de Variables de Entorno ---
# NOTA: Hugging Face Spaces debe tener la variable OPENAI_API_KEY en los Secrets.
load_dotenv() 
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- 2. Rutas (Relativas a la raíz del proyecto) ---
MODEL_PATH = "models/hypertension_model.joblib"
FAISS_PATH = "models/faiss_index"

# --- 3. Carga Global de Modelos (Se ejecuta al importar el archivo UNA SOLA VEZ) ---
ml_model = None
rag_chain = None

# Cargar Cerebro 1: Modelo ML (El Analista)
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo ML (Analista) cargado desde {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR al cargar modelo ML: {e}")

# Cargar Cerebro 2: Sistema RAG (El Coach)
try:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY no encontrada. Configura los Secrets de Hugging Face.")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"Índice FAISS no encontrado en {FAISS_PATH}.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # IMPORTANTE: Necesario para cargar FAISS de forma segura
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever() 

    # --- PROMPT Y CADENA (COPIADO EXACTO DE api/main.py) ---
    prompt_template = """
    **Mi Identidad (Persona):**
    Eres "NexusByte", un coach de salud IA. Tu tono es amigable, profesional y empático.

    **Mi Base de Conocimiento (Contexto):**
    Mi conocimiento se limita *estrictamente* a la información proporcionada en el "Contexto" de abajo.
    {context}

    **Mis Reglas de Operación (¡Muy Importante!):**
    1. Revisa el "Historial del Chat". Úsalo para entender preguntas de seguimiento (ej. "cómo hago eso", "por qué").
    2. Si el "Humano" solo saluda, responde al saludo amigablemente.
    3. Si el "Humano" pregunta algo "En Contexto", usa la info del Contexto para responder.
    4. Si el "Humano" pregunta por "sus resultados" o "su riesgo", indícale que use el "Calculador de Riesgo (ML)" de la app.
    5. Si la respuesta NO está en el "Contexto" O en el "Historial", di amablemente que no tienes esa información.
    
    ---
    **Historial del Chat (para darte contexto):**
    {chat_history}
    ---
    
    **Información de mi Base de Conocimiento (Contexto):**
    {context}
    
    **Nueva Pregunta del Humano:**
    {question}

    NexusByte (Respuesta en español):
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("history") 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Sistema RAG (Coach) cargado exitosamente.")

except Exception as e:
    rag_chain = None
    print(f"❌ ERROR al cargar sistema RAG: {e}. Revisa claves y rutas.")


# --- 4. Funciones de Inferencia (Reemplazan Endpoints de la API) ---

def get_risk_prediction(data: dict) -> dict:
    """Reemplaza el endpoint /predict."""
    if ml_model is None:
        return {"risk_score": 0.5, "prediction": 0, "error": "Modelo ML no está cargado."}
    
    try:
        # data ya es el diccionario que viene de Streamlit
        input_data = pd.DataFrame([data])
        probabilities = ml_model.predict_proba(input_data)[0]
        prediction = ml_model.predict(input_data)[0]
        risk_score = float(probabilities[1])
        
        # Debes añadir los 'drivers' si tu modelo los genera (ej. con SHAP)
        return {"risk_score": risk_score, "prediction": int(prediction)}
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"risk_score": 0.5, "prediction": 0, "error": f"Error en predicción: {e}"}

def get_coach_plan(prediction_output: dict, features: dict) -> str:
    """Reemplaza el endpoint /coach y utiliza el RAG."""
    if rag_chain is None:
        return "ERROR: Sistema RAG no está disponible para generar el plan."
    
    try:
        data = prediction_output
        
        # Lógica copiada de /coach en api/main.py
        if data.get("prediction", 0) == 1:
            riesgo_desc = f"Mi riesgo de hipertensión es alto (score: {data.get('risk_score', 0):.2f})."
        else:
            riesgo_desc = f"Mi riesgo de hipertensión es bajo (score: {data.get('risk_score', 0):.2f})."
        
        # Esta pregunta es crucial para que el RAG te dé el plan inicial
        pregunta_usuario = f"{riesgo_desc} ¿Qué consejos de salud (nutrición, ejercicio, estrés) me puedes dar basándote en tu conocimiento?"
        
        coach_message = rag_chain.invoke({
            "question": pregunta_usuario,
            "history": "" 
        })
        
        return coach_message
        
    except Exception as e:
        print(f"Error en el RAG chain para el plan: {e}")
        return f"ERROR en el RAG chain para el plan: {e}"

def get_chat_response(query: str, history: list) -> str:
    """Reemplaza el endpoint /chat y utiliza el RAG con memoria."""
    if rag_chain is None:
        return "ERROR: Sistema RAG no está disponible para responder el chat."
    
    try:
        # Formatear el historial igual que en api/main.py
        history_formatted = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        response = rag_chain.invoke({
            "question": query,
            "history": history_formatted
        })
        return response
    except Exception as e:
        print(f"Error en el RAG chain para el chat: {e}")
        return f"ERROR en el RAG chain para el chat: {e}"