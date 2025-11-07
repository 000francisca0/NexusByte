import joblib
import pandas as pd
import numpy as np
import os

# Importaciones del Sistema RAG (LangChain/LLM)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter  # ESTO ES LO QUE FALLA AHORA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. CONFIGURACIÓN ---

# Verifica que tu clave de OpenAI esté configurada
if "OPENAI_API_KEY" not in os.environ:
    print("ADVERTENCIA: La variable de entorno OPENAI_API_KEY no está configurada. El sistema RAG fallará.")


# --- 2. MODELO ML: CARGA Y PREDICCIÓN ---

def load_ml_model(model_path: str):
    """Carga el modelo ML (Joblib) desde la ruta especificada."""
    try:
        # Resuelve la ruta relativa (sube de src/ a la raíz y busca en models/)
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        full_path = os.path.join(base_dir, model_path)
        
        # Carga el modelo
        model = joblib.load(full_path)
        print(f"✅ Modelo ML cargado desde: {full_path}")
        return model
    except Exception as e:
        print(f"❌ Error al cargar el modelo ML en {full_path}: {e}")
        return None

def get_risk_score(ml_model, user_data: dict) -> tuple:
    """Calcula el score de riesgo y devuelve los principales factores que contribuyen."""
    if ml_model is None:
        return 0.0, ["Error: Modelo ML no disponible."]

    # 2.1 Preprocesamiento de datos para el modelo
    
    # Calcular IMC (BMI)
    height_m = user_data['height_cm'] / 100.0
    bmi = user_data['weight_kg'] / (height_m ** 2)

    # 1 para Femenino, 0 para Masculino
    sex_binary = 1 if user_data['sex_code'] == 'F' else 0

    # DataFrame con las 9 columnas que el modelo espera
    input_data = pd.DataFrame({
        'Age_Years_mean': [user_data['age']],
        'Sex_Female': [sex_binary],  
        'BMI_mean': [bmi],
        'Waist_Circumference_cm_mean': [user_data['waist_cm']],
        'Hours_Sleep_mean': [user_data['sleep_hours']],
        'Cigarettes_per_day_mean': [user_data['smokes_cig_day']],
        'Days_MVPA_per_week_mean': [user_data['days_mvpa_week']],
        'Fruit_Vegetable_Portions_day_mean': [user_data['fruit_veg_portions_day']],
        'Height_cm_mean': [user_data['height_cm']] 
    })
    
    # 2.2 Predicción
    try:
        # La probabilidad de la clase 1 (riesgo)
        risk_score = ml_model.predict_proba(input_data)[:, 1][0] 
    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}")
        return 0.0, ["Error en la predicción."]
    
    # 2.3 Extracción de Drivers (Factores Clave - Lógica heurística)
    drivers = []
    
    if bmi > 30: drivers.append("IMC elevado (Obesidad)")
    elif bmi > 25: drivers.append("Sobrepeso (IMC)")
    
    # Puntos de corte estándar OMS simplificados: M > 90cm, F > 80cm
    if user_data['waist_cm'] > (90 if user_data['sex_code'] == 'M' else 80): 
        drivers.append("Circunferencia de cintura alta (Riesgo Abdominal)")
        
    if user_data['smokes_cig_day'] > 0: drivers.append("Consumo de tabaco")
    
    if user_data['days_mvpa_week'] < 3: drivers.append("Baja actividad física vigorosa")
    
    if user_data['fruit_veg_portions_day'] < 5: drivers.append("Baja ingesta de frutas/verduras")
    
    if user_data['sleep_hours'] < 6.5 or user_data['sleep_hours'] > 8.5: drivers.append("Patrón de sueño inadecuado")
    
    if not drivers:
        drivers = ["Edad", "Circunferencia de cintura", "IMC"]

    unique_drivers = list(set(drivers))
    return risk_score, unique_drivers[:4]


# --- 3. SISTEMA RAG: CARGA Y CONSULTA (EN MEMORIA) ---

# BASE DE CONOCIMIENTO (mini-KB) - Integrada en el código
KB_TEXT = """
Tópicos de Bienestar y Salud Preventiva (Mini-KB para RAG):

1. Nutrición: Dieta mediterránea: rica en frutas, verduras, granos integrales, aceite de oliva y proteínas magras. Reducir la sal (menos de 5g/día), azúcares añadidos y grasas saturadas es clave. Mínimo de 5 porciones de frutas y verduras al día.
2. Actividad Física: 150 minutos de actividad aeróbica moderada O 75 minutos vigorosa por semana, más 2 días de entrenamiento de fuerza. El sedentarismo prolongado aumenta el riesgo.
3. Sueño y Descanso: Adultos deben apuntar a 7-9 horas de sueño de calidad. El sueño insuficiente eleva cortisol y afecta metabolismo e inflamación.
4. Estrés y Manejo Emocional: Técnicas de mindfulness, meditación y manejo del tiempo. El estrés crónico contribuye a la hipertensión y riesgo cardiometabólico.
5. Prevención Cardiovascular: Control de peso (IMC < 25), presión arterial (< 120/80 mmHg ideal), y colesterol. La grasa abdominal (cintura alta) es un indicador de riesgo superior al IMC en algunos casos.
6. Tabaquismo: Dejar de fumar reduce el riesgo a la mitad en un año. Es el factor modificable más crítico para enfermedad cardiovascular.

REGLAS CLAVE PARA EL COACH:
- No diagnosticar enfermedades.
- Siempre referir a un profesional de la salud (médico, nutricionista o kinesiólogo) para decisiones finales.
- El plan de acción debe ser motivador, realista y tener una duración de 2 semanas.
- El tono debe ser amable, alentador y empático (estilo "coach").
"""

def load_rag_system():
    """Carga la Base de Conocimiento en memoria y configura la cadena RAG."""
    try:
        # 3.1 Inicializar Embeddings y LLM
        embeddings = OpenAIEmbeddings()
        llm = OpenAI(temperature=0.0) 

        # 3.2 Creación del Vector Store EN MEMORIA (In-Memory)
        # Esto reemplaza a FAISS.load_local() y evita los errores de archivos faltantes
        texts = KB_TEXT.split('\n')
        clean_texts = [t for t in texts if t.strip()] 
        
        # Crea la base de datos FAISS en memoria y el retriever
        vector_store = FAISS.from_texts(clean_texts, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 3.3 Definición del Prompt RAG
        rag_prompt_template = PromptTemplate(
            template="""Eres un Coach de Bienestar Preventivo en Chile. Tu tono es motivador y empático.

            [REGLAS CLAVE: 1. No diagnosticar. 2. Siempre referir al profesional. 3. Usar el contexto proporcionado.]

            Contexto de la Base de Conocimiento (mini-KB):
            {context}

            Instrucciones Específicas del Usuario:
            {question}

            1. Si el usuario pide un "plan", genera un Plan de Acción de 2 Semanas conciso y en formato de lista.
            2. Si es una pregunta, responde con consejos relevantes basados en el contexto.
            3. Tu respuesta debe ser concisa y útil, no repitas la pregunta.
            """,
            input_variables=["context", "question"]
        )

        # 3.4 Configuración de la cadena RAG completa
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt_template
            | llm
            | StrOutputParser()
        )
        
        print(f"✅ Sistema RAG cargado y configurado en memoria.")
        return rag_chain

    except Exception as e:
        print(f"❌ Error al cargar el sistema RAG: {e}")
        print("Asegúrate de que la clave OPENAI_API_KEY esté disponible.")
        return None

def generate_rag_response(rag_system, prompt: str) -> tuple:
    """
    Genera la respuesta del LLM utilizando el sistema RAG.
    Devuelve (respuesta_texto, es_plan_generado_booleano).
    """
    if rag_system is None:
        return "Lo siento, el Coach IA no está disponible.", False
    
    # 4.1 Generación de la Respuesta
    try:
        response = rag_system.invoke(prompt)
        
        # 4.2 Detección de la Intención (¿Se generó un plan?)
        plan_keywords = ["plan", "semanas", "acción", "rutina", "cronograma"]
        is_plan_generated = any(keyword in prompt.lower() for keyword in plan_keywords)

        # 4.3 Añadir el Disclaimer Final
        disclaimer = "\n\n---\n*Recuerda: Soy un coach IA. Consulta siempre con un profesional de la salud.*"
        
        return response + disclaimer, is_plan_generated
    
    except Exception as e:
        print(f"Error al invocar la cadena RAG: {e}")
        return "Hubo un error al comunicarme con el Coach IA. Por favor, revisa tu clave de API o la conexión a internet.", False