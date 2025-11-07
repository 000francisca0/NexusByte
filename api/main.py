# Contenido para: api/main.py (Versión 2.3.1 - ¡CON MEMORIA!)

import uvicorn
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from operator import itemgetter # ¡NUEVA IMPORTACIÓN!

# Importaciones de Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Carga de .env y Aplicación ---
load_dotenv() 
app = FastAPI(
    title="API Coach de Bienestar Hídrido (ML + RAG)",
    version="2.3.1",
    description="Predice riesgo (ML) y chatea sobre consejos (RAG)."
)

# --- 2. Carga de Modelos (ML y RAG) ---
MODEL_PATH = "models/hypertension_model.joblib"
FAISS_PATH = "models/faiss_index"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Cargar Cerebro 1: Modelo ML (El Analista)
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"Modelo ML (Analista) cargado desde {MODEL_PATH}")
except Exception as e:
    ml_model = None
    print(f"ERROR al cargar modelo ML: {e}")

# Cargar Cerebro 2: Sistema RAG (El Coach)
try:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY no encontrada. Revisa tu .env")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"Índice FAISS no encontrado en {FAISS_PATH}.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever() 

    # --- ¡PROMPT CON MEMORIA! ---
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
    
    # --- ¡CADENA CON MEMORIA! ---
    # Reconfiguramos la cadena para aceptar 'question' e 'history'
    rag_chain = (
        {
            "context": itemgetter("question") | retriever, # El retriever sigue buscando solo con la última pregunta
            "question": itemgetter("question"),
            "chat_history": itemgetter("history") # Pasamos el historial al prompt
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Sistema RAG (Coach v2.3.1 con Memoria) cargado exitosamente.")

except Exception as e:
    rag_chain = None
    print(f"ERROR al cargar sistema RAG: {e}")


# --- 3. Modelos de Datos (Pydantic) ---
class FeaturesInput(BaseModel):
    feat_imc: float = Field(..., json_schema_extra={'example': 28.5})
    feat_whtr: float = Field(..., json_schema_extra={'example': 0.55})
    feat_age: int = Field(..., json_schema_extra={'example': 45})
    feat_sex: int = Field(..., json_schema_extra={'example': 0}, description="0: Hombre, 1: Mujer")
    feat_is_smoker: int = Field(..., json_schema_extra={'example': 1}, description="0: No, 1: Sí")
    feat_sleep_hours: float = Field(..., json_schema_extra={'example': 6.5})
    feat_activity_days: float = Field(..., json_schema_extra={'example': 2.0})

class PredictionOutput(BaseModel):
    risk_score: float
    prediction: int

# --- ¡MODELO DE CHAT CON MEMORIA! ---
class ChatInput(BaseModel):
    query: str = Field(..., example="¿Qué es la dieta DASH?")
    history: list = Field(default_factory=list, example=[{"role": "user", "content": "Hola"}])


# --- 4. Endpoints de la API ---
@app.get("/")
def read_root():
    return {"status": "API Híbrida (ML + RAG) está en línea."}

@app.post("/predict", response_model=PredictionOutput)
def predict_hypertension(data: FeaturesInput):
    # (Cerebro 1: El Analista ML)
    if ml_model is None:
        raise HTTPException(status_code=500, detail="Modelo ML no está cargado.")
    try:
        input_data = pd.DataFrame([data.dict()])
        probabilities = ml_model.predict_proba(input_data)[0]
        prediction = ml_model.predict(input_data)[0]
        risk_score = float(probabilities[1])
        return PredictionOutput(risk_score=risk_score, prediction=int(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

@app.post("/coach")
def get_coaching_advice(data: PredictionOutput):
    # (Cerebro 2: El Coach RAG para Consejo Específico)
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Sistema RAG (Coach) no está cargado.")
    try:
        if data.prediction == 1:
            riesgo_desc = f"Mi riesgo de hipertensión es alto (score: {data.risk_score:.2f})."
        else:
            riesgo_desc = f"Mi riesgo de hipertensión es bajo (score: {data.risk_score:.2f})."
        
        pregunta_usuario = f"{riesgo_desc} ¿Qué consejos de salud (nutrición, ejercicio, estrés) me puedes dar basándote en tu conocimiento?"
        coach_message = rag_chain.invoke({
            "question": pregunta_usuario,
            "history": "" # El coach inicial no tiene historial
        })

        return {"user_risk_score": data.risk_score, "coach_message": coach_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el RAG chain: {e}")

# --- ¡ENDPOINT DE CHAT CON MEMORIA! ---
@app.post("/chat")
def handle_chat_query(data: ChatInput):
    """Cerebro 3: El Chatbot General (RAG)"""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Sistema RAG (Coach) no está cargado.")
    try:
        # Formatear el historial para que sea un texto simple
        history_formatted = "\n".join([f"{msg['role']}: {msg['content']}" for msg in data.history])
        
        response = rag_chain.invoke({
            "question": data.query,
            "history": history_formatted
        })
        return {"user_query": data.query, "coach_message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el RAG chain: {e}")

# --- 5. Ejecución ---
if __name__ == "__main__":
    print("Iniciando servidor Uvicorn en http://127.0.0.1:8000")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)