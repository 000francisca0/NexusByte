import os
import pandas as pd
import joblib
from pydantic import ValidationError
from typing import Dict, Any, Optional, Tuple, List

# Dependencias de LangChain y OpenAI (usando las nuevas importaciones robustas)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# Importación CORREGIDA y compatible para el text splitter
from langchain_text_splitters import CharacterTextSplitter 

# Importación local de prompts (asumiendo que existen y tienen las variables esperadas)
try:
    from src.prompts import RAG_PROMPT_TEMPLATE, JSON_PROMPT_TEMPLATE
except ImportError:
    # Definiciones de fallback si prompts.py no existe o falla
    RAG_PROMPT_TEMPLATE = (
        "Eres un Coach de Bienestar Preventivo IA. "
        "Basado en el contexto proporcionado y los siguientes factores de riesgo ({risk_drivers}), "
        "genera un plan de acción de 2 semanas conciso y motivador. "
        "Contexto de conocimiento: {rag_context}\n\n"
        "Plan para los factores: {risk_drivers}"
    )
    JSON_PROMPT_TEMPLATE = "Parse the following text into a JSON object matching the schema: {json_schema}. Text: {profile_text}"


# --- CONSTANTES Y CONFIGURACIÓN ---

# CRÍTICO: Se intenta leer la clave desde el entorno. Si no está, será None.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

MODEL_PATH = "models/hypertension_model.joblib"
RAG_DATA_PATH = "data/Dietary_TRAIN.csv" # Usamos el CSV simple para el RAG
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"

# --- RAG SYSTEM (In-Memory) ---
def load_rag_system(rag_data_path: str = RAG_DATA_PATH) -> Tuple[Any, Any]:
    """
    Carga y prepara el sistema RAG (Retrieval Augmented Generation) 
    en memoria usando LangChain.
    """
    
    # CRÍTICO: TRUCO FINAL para asegurar la lectura de la clave
    api_key_value = os.environ.get("OPENAI_API_KEY")

    if not api_key_value:
        print("ADVERTENCIA: OPENAI_API_KEY no se pudo leer desde el entorno. El RAG fallará.")
        return None, None
    
    try:
        # 1. Cargar datos simulados como texto
        df_rag = pd.read_csv(rag_data_path)
        rag_text = df_rag.to_string(index=False)

        # 2. Splitter 
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.create_documents([rag_text])

        # 3. Embeddings (usando la clave leída explícitamente)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key_value, model=EMBEDDING_MODEL)

        # 4. Vector Store (Chroma en memoria)
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        # 5. LLM (usando la clave leída explícitamente)
        llm = ChatOpenAI(openai_api_key=api_key_value, model=LLM_MODEL, temperature=0.7)

        # 6. Retorno de Retriever y LLM
        retriever = vectorstore.as_retriever()
        
        print("Sistema RAG cargado exitosamente.")
        return retriever, llm

    except Exception as e:
        # Si esto falla, es 99% la clave (fondos, validez, permisos)
        print(f"ERROR CRÍTICO FINAL AL INICIALIZAR RAG. Mensaje de LangChain/OpenAI: {e}")
        return None, None


# --- ML MODEL ---
def load_ml_model(model_path: str = MODEL_PATH) -> Any:
    """Carga el modelo de Machine Learning desde la ruta especificada."""
    try:
        ml_model = joblib.load(model_path)
        return ml_model
    except Exception as e:
        print(f"Error al cargar el modelo ML: {e}")
        return None

def get_risk_score(ml_model: Any, profile_data: Dict[str, Any], feature_cols: list) -> float:
    """Calcula el score de riesgo de hipertensión."""
    if ml_model is None:
        return -1.0 # Indica error o modelo no cargado
    
    try:
        # Crear DataFrame con los datos del perfil y asegurar el orden de las features
        data_for_prediction = pd.DataFrame([profile_data], columns=feature_cols)
        data_for_prediction = data_for_prediction[feature_cols]

        # Predicción (usando predict_proba)
        risk_score = ml_model.predict_proba(data_for_prediction)[:, 1][0]
        
        return float(risk_score)

    except Exception as e:
        print(f"Error al calcular el score de riesgo: {e}")
        return -1.0 # Indica error


# --- GENERACIÓN RAG ---
def generate_rag_response(llm: Any, retriever: Any, user_query: str) -> str:
    """
    Genera el plan de acción usando RAG.
    
    Argumentos:
        llm: El LLM cargado.
        retriever: El retriever de la vector store.
        user_query: La consulta completa (perfil + pregunta) que viene de app/app.py.
    """
    if llm is None or retriever is None:
        return "El sistema RAG no está disponible. No se pueden generar recomendaciones. Revisa tu clave de OpenAI."

    try:
        # 1. Búsqueda (Retrieval)
        relevant_docs = retriever.get_relevant_documents(user_query)
        rag_context = "\n---\n".join([doc.page_content for doc in relevant_docs])

        # 2. Generación (Augmentation)
        # Usamos la user_query como el principal 'risk_drivers' en el template para darle contexto completo.
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(
            risk_drivers=user_query, 
            rag_context=rag_context
        )

        response = llm.invoke(prompt).content
        return response

    except Exception as e:
        print(f"Error durante la generación RAG: {e}")
        return f"Error en la generación de la respuesta. Causa: {e}"


# --- DATOS Y PARSER (Función auxiliar, no crítica para el flujo principal) ---
def parse_profile_to_json(llm: Any, profile_text: str, json_schema: str) -> Optional[Dict[str, Any]]:
    """Usa el LLM para parsear texto de perfil a un objeto JSON."""
    if llm is None:
        return None

    try:
        prompt = PromptTemplate.from_template(JSON_PROMPT_TEMPLATE).format(
            profile_text=profile_text,
            json_schema=json_schema
        )
        
        response = llm.invoke(prompt).content
        import json
        return json.loads(response)

    except Exception as e:
        print(f"Error en el parsing LLM o JSON: {e}")
        return None