
Markdown

# NexusByte: Coach de Bienestar Híbrido (ML + RAG)

![Versión](https://img.shields.io/badge/version-2.3.1-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-yellow.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-FastAPI_&_Streamlit-green.svg)
![AI](https://img.shields.io/badge/AI-RAG_&_Scikit--learn-orange.svg)

Prototipo de una aplicación de salud híbrida que combina Machine Learning (ML) para análisis predictivo y Generación Aumentada por Recuperación (RAG) para coaching conversacional personalizado, cumpliendo con la pauta del 1° Hackathon de IA Aplicada Duoc UC 2025.

---

## 🚀 Guía Rápida de Instalación y Uso (Local)

Para ejecutar este proyecto, necesitarás Python 3.11 instalado. La aplicación funciona en dos partes: el "Cerebro" (la API) y la "Cara" (la App web). Necesitarás dos terminales abiertas.

### Paso 1: Clonar (descargar) el Proyecto
Abre tu terminal (PowerShell en Windows, Terminal en Kali) y clona el repositorio.
```bash
git clone [https://github.com/000francisca0/NexusByte.git](https://github.com/000francisca0/NexusByte.git)
cd NexusByte
Paso 2: Configurar el Entorno Virtual
Vamos a crear un entorno virtual (venv) para instalar las librerías de forma segura.

En Windows (PowerShell):

PowerShell

# 1. Crea el entorno
python -m venv venv

# 2. Activa el entorno
.\venv\Scripts\Activate.ps1
(Si ves un error rojo sobre "ejecución de scripts", ejecuta Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process y vuelve a intentarlo).

En Kali (Linux):

Bash

# 1. Crea el entorno
python3 -m venv venv

# 2. Activa el entorno
source venv/bin/activate
Paso 3: Instalar las Librerías
Una vez que veas (venv) al inicio de tu terminal, instala todo lo necesario:

Bash

pip install -r requirements.txt
Paso 4: Configurar los Secretos (¡Importante!)
Crea un archivo nuevo llamado .env en la raíz del proyecto (NexusByte/). Ábrelo y pega esta línea, reemplazando con tu propia llave de API de OpenAI:

OPENAI_API_KEY="sk-TU_LLAVE_SECRETA_DE_OPENAI_VA_AQUI"
Paso 5: Iniciar el "Cerebro" (Terminal 1)
¡Es hora de arrancar! En tu terminal actual, inicia el servidor de la API:

Bash

uvicorn api.main:app
(No usamos --reload, ya que es para modo local estable). Déjala abierta. Verás que carga los modelos (.joblib y faiss_index) exitosamente.

Paso 6: Iniciar la "Cara" (Terminal 2)
Abre una NUEVA terminal (¡deja la primera corriendo!).

Navega a la carpeta (cd NexusByte).

Activa el entorno virtual (.\venv\Scripts\Activate.ps1 o source venv/bin/activate).

Ejecuta la aplicación web de Streamlit:

Bash

streamlit run app/app.py
