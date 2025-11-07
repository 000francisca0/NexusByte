import os
import pandas as pd
import joblib
from pydantic import ValidationError
from typing import Dict, Any, Optional, Tuple

# Importaciones específicas de LangChain (actualizadas para el error de HFS)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
# CORRECCIÓN CLAVE: Importamos desde el paquete específico 'langchain_text_splitters'
from langchain_text_splitters import CharacterTextSplitter 

# Importación local de prompts
from src.prompts import RAG_PROMPT_TEMPLATE, JSON_PROMPT_TEMPLATE

# --- CONSTANTES ---
# La clave se cargará automáticamente desde los Secrets de Hugging Face
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
MODEL_PATH = "models/hypertension_model.joblib"
# Usamos un archivo de datos .csv sencillo para evitar la complejidad de PDF en este entorno
RAG_DATA_PATH = "data/Dietary_TRAIN.csv" 
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"

# --- RAG SYSTEM (In-Memory) ---
def load_rag_system(rag_data_path: str = RAG_DATA_PATH) -> Tuple[Any, Any]:
    """
    Carga y prepara el sistema RAG (Retrieval Augmented Generation) 
    en memoria usando LangChain para generar recomendaciones.
    """
    if not OPENAI_API_KEY:
        print("ADVERTENCIA: OPENAI_API_KEY no está configurada. El sistema RAG no funcionará.")
        return None, None
    
    try:
        # Cargar datos simulados como texto para el RAG
        df_rag = pd.read_csv(rag_data_path)
        rag_text = df_rag.to_string(index=False)
        documents = [rag_text] # Tratamos el CSV como un solo documento grande de texto

        # 1. Splitter (ahora importado desde langchain_text_splitters)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 2. División de documentos (simulando los chunks)
        texts = text_splitter.create_documents([rag_text])

        # 3. Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

        # 4. Vector Store
        # Usamos un almacenamiento temporal en memoria (no persistente)
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        # 5. LLM
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.7)

        # 6. Retorno de Retriever y LLM
        retriever = vectorstore.as_retriever()
        
        return retriever, llm

    except Exception as e:
        print(f"Error al cargar el sistema RAG: {e}")
        return None, None

# --- ML MODEL ---
def load_ml_model(model_path: str = MODEL_PATH) -> Any:
    """Carga el modelo de Machine Learning desde la ruta especificada."""
    try:
        # La ruta es relativa a la raíz del proyecto (NexusByte/)
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
        # 1. Crear DataFrame con los datos del perfil
        data_for_prediction = pd.DataFrame([profile_data], columns=feature_cols)
        
        # 2. Asegurarse de que las columnas están en orden (IMPORTANTE)
        # Esto asume que 'feature_cols' es la lista correcta de características esperadas por el modelo
        data_for_prediction = data_for_prediction[feature_cols]

        # 3. Predicción (usando predict_proba para obtener el riesgo de la clase positiva)
        # Asume que la clase positiva (riesgo) está en el índice 1
        risk_score = ml_model.predict_proba(data_for_prediction)[:, 1][0]
        
        return float(risk_score)

    except Exception as e:
        print(f"Error al calcular el score de riesgo: {e}")
        return -1.0 # Indica error

# --- DATOS Y PARSER (Se mantiene, aunque no se usa en app/app.py) ---
def parse_profile_to_json(llm: Any, profile_text: str, json_schema: str) -> Optional[Dict[str, Any]]:
    """
    Usa el LLM para parsear texto de perfil a un objeto JSON que cumple con el esquema.
    """
    if llm is None:
        print("Error: LLM no disponible para parsing.")
        return None

    try:
        # Crear el prompt específico para el parser
        prompt = PromptTemplate.from_template(JSON_PROMPT_TEMPLATE).format(
            profile_text=profile_text,
            json_schema=json_schema
        )
        
        # Llamada al LLM
        response = llm.invoke(prompt).content
        
        # Intenta cargar la respuesta como JSON
        import json
        return json.loads(response)

    except Exception as e:
        print(f"Error en el parsing LLM o JSON: {e}")
        return None

# --- GENERACIÓN RAG (CORRECCIÓN CLAVE) ---
def generate_rag_response(llm: Any, retriever: Any, user_query: str) -> str:
    """
    Genera el plan de acción usando RAG.
    
    Argumentos:
        llm: El LLM cargado (ChatOpenAI).
        retriever: El retriever de la vector store (Chroma).
        user_query: La consulta completa que viene de app/app.py, incluyendo el perfil y la pregunta.
    """
    if llm is None or retriever is None:
        return "El sistema RAG no está disponible. No se pueden generar recomendaciones."

    try:
        # 1. Búsqueda (Retrieval)
        # Usamos la consulta del usuario para buscar documentos relevantes.
        relevant_docs = retriever.get_relevant_documents(user_query)
        rag_context = "\n---\n".join([doc.page_content for doc in relevant_docs])

        # 2. Generación (Augmentation)
        # Formateamos el prompt con el contexto encontrado y la consulta del usuario.
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(
            risk_drivers=user_query, # Aunque el nombre del template es risk_drivers, le pasamos la query completa
            rag_context=rag_context
        )

        response = llm.invoke(prompt).content
        return response

    except Exception as e:
        print(f"Error durante la generación RAG: {e}")
        return "Error en la generación de la respuesta. Verifica la configuración de OpenAI."