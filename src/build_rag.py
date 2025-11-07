# Contenido para: src/build_rag.py (VERSIÓN CORREGIDA)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Configuración de Rutas (MODO INTELIGENTE) ---

# 1. Obtener la ruta absoluta del directorio donde está este script (NEXUSBYTE/src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Subir un nivel para estar en la raíz del proyecto (NEXUSBYTE/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 3. Construir las rutas a 'data/kb' y 'models/' desde la raíz
KB_DIR = os.path.join(PROJECT_ROOT, "data/kb")
FAISS_PATH = os.path.join(PROJECT_ROOT, "models/faiss_index") # El "cerebro" cocinado se guarda aquí

# 4. Cargar el .env desde la raíz del proyecto
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def build_vector_store():
    print("--- Iniciando construcción del Vector Store (RAG) ---")
    
    # 1. Cargar la API Key de OpenAI (de forma segura desde .env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"ERROR: OPENAI_API_KEY no encontrada en {os.path.join(PROJECT_ROOT, '.env')}")
        print("Asegúrate de tener un archivo .env en la raíz del proyecto.")
        return

    # 2. Cargar los documentos de la Base de Conocimiento
    print(f"Cargando documentos desde {KB_DIR}...")
    if not os.path.exists(KB_DIR):
        print(f"ERROR: La ruta de KB no existe: {KB_DIR}")
        return
        
    loader = DirectoryLoader(
        KB_DIR, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()
    if not docs:
        print(f"ERROR: No se encontraron archivos .txt en {KB_DIR}")
        return
    print(f"Se encontraron {len(docs)} documentos.")

    # 3. Dividir los documentos en "chunks" (trozos)
    print("Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # 4. Crear los "Embeddings" (vectores) y FAISS
    # ¡ESTE PASO USA TU API KEY Y CUESTA DINERO! (muy poco)
    print("Creando embeddings con OpenAI y guardando en FAISS...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 5. Guardar el índice de FAISS localmente
    vectorstore.save_local(FAISS_PATH)
    
    print(f"\n✅ ¡Éxito! Vector Store guardado en: {FAISS_PATH}")

if __name__ == "__main__":
    build_vector_store()