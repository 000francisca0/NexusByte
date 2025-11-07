# Contenido para: api/main.py (VERSIÓN 2.0.2 - LA FINAL)

import uvicorn
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importaciones de Langchain (¡LAS BUENAS!)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Carga de .env y Aplicación ---
load_dotenv() 
app = FastAPI(
    title="API Coach de Bienestar Híbrido (ML + RAG)",
    version="2.0.2",
    description="Predice riesgo de hipertensión (ML) y da consejos (RAG)."
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
        raise FileNotFoundError(f"Índice FAISS no encontrado en {FAISS_PATH}. Corre src/build_rag.py primero.")
    
    # Inicializa los componentes RAG
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=openai_api_key)
    
    # Este es el "recuperador" que busca en tu base de conocimiento
    retriever = vectorstore.as_retriever() 

    # Plantilla de Prompt (El "cerebro" del coach)
    prompt_template = """
    Eres un coach de salud preventivo. Tu tono es motivador y empático.
    Usa el siguiente contexto de tu base de conocimiento para responder.
    Contexto: {context}
    Pregunta del Usuario: {question}
    Respuesta del Coach (en español):
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # --- ¡LA NUEVA LÓGICA! (La "tubería" LCEL) ---
    # Esto reemplaza al 'RetrievalQA.from_chain_type' que estaba roto.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # -----------------------------------------------
    
    print("Sistema RAG (Coach) cargado exitosamente (¡con LCEL!).")

except Exception as e:
    rag_chain = None
    print(f"ERROR al cargar sistema RAG: {e}")


# --- 3. Modelos de Datos (Pydantic) ---
# (CORREGIDO para eliminar los warnings de 'example=')
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

# --- 4. Endpoints de la API ---
@app.get("/")
def read_root():
    return {"status": "API Híbrida (ML + RAG) está en línea."}

@app.post("/predict", response_model=PredictionOutput)
def predict_hypertension(data: FeaturesInput):
    """Cerebro 1: El Analista (ML)"""
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
    """Cerebro 2: El Coach (RAG + LLM)"""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Sistema RAG (Coach) no está cargado.")
    try:
        if data.prediction == 1:
            riesgo_desc = f"Mi riesgo de hipertensión es alto (score: {data.risk_score:.2f})."
        else:
            riesgo_desc = f"Mi riesgo de hipertensión es bajo (score: {data.risk_score:.2f})."
        
        pregunta_usuario = f"{riesgo_desc} ¿Qué consejos de salud (nutrición, ejercicio, estrés) me puedes dar basándote en tu conocimiento?"
        
        # ¡LA NUEVA LÓGICA! Usamos .invoke() en la tubería
        coach_message = rag_chain.invoke(pregunta_usuario)

        return {"user_risk_score": data.risk_score, "coach_message": coach_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el RAG chain: {e}")

# --- 5. Ejecución ---
if __name__ == "__main__":
    print("Iniciando servidor Uvicorn en http://127.0.0.1:8000")
    # (CORREGIDO para eliminar el 'reload=True' que rompía el 'SpawnProcess')
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)