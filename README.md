Coach de Bienestar NexusByte
Descripción General

NexusByte es un Coach de Bienestar Híbrido desarrollado como parte del Desafío Salud NHANES 2025 – Duoc UC.
Integra Machine Learning (ML), procesamiento de lenguaje natural (IA conversacional) y una interfaz intuitiva en Streamlit, para ofrecer:

-Evaluación de riesgo de hipertensión arterial mediante un modelo ML entrenado con datos NHANES.
-Recomendaciones personalizadas generadas por IA (endpoint /coach).
-Chat interactivo (RAG) para educación en hábitos saludables.
- Reportes descargables en PDF con plan de acción y disclaimer médico.

Arquitectura General del Proyecto.
Proyecto_NexusByte/
├── api/                     # API FastAPI (endpoints /predict, /chat, /coach)
│   └── main.py
├── app/
│   ├── app.py               # Interfaz Streamlit (versión 5.6)
│   ├── coach_image.png      # Avatar del coach (imagen circular animada)
│   ├── results.csv          # Registro de resultados ML
│   └── logs.jsonl           # Bitácora de interacciones del chat
├── src/
│   ├── model_train.py       # Entrenamiento del modelo ML (XGBoost / sklearn)
│   └── model.pkl            # Modelo serializado
├── kb/                      # Base de conocimiento para el chat RAG
│   ├── nutricion.md
│   ├── actividad_fisica.md
│   └── sueno_saludable.md
├── requirements.txt         # Dependencias
└── README.md                # Documentación principal

*Instalación y Ejecución*

*Crear entorno virtual*
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

*Instalar dependencias*
pip install -r requirements.txt

*Ejecutar el servidor FastAPI*
uvicorn api.main:app --reload

*Ejecutar la aplicación Streamlit*
streamlit run app/app.py

Luego abre el navegador en:
http://localhost:8501


*Modelo de Machine Learning*
| Característica  | Descripción                                                                |
| --------------- | -------------------------------------------------------------------------- |
| **Target:**     | Riesgo de hipertensión (binario: 0 = bajo, 1 = alto)                       |
| **Datos:**      | NHANES 2007–2020 (submuestra adultos 18–80 años)                           |
| **Variables:**  | Edad, sexo, IMC, cintura/talla (WHtR), tabaquismo, sueño, actividad física |
| **Modelo:**     | XGBoost / Logistic Regression                                              |
| **Validación:** | Temporal holdout (2007–2016 → train, 2017–2020 → test)                     |
| **Métricas:**   | AUROC = 0.82, Brier = 0.12, Fairness gap (sexo) = 0.03                     |

*IA Conversacional (RAG)*
El módulo /chat implementa un asistente educativo que responde preguntas de bienestar basándose en una mini base de conocimiento (kb/) con embeddings o BM25.
Ejemplos:
“¿Qué es la dieta DASH?”
“¿Cuántas horas de sueño recomiendan los expertos?”
“¿Cómo puedo reducir el estrés diario?”
Toda interacción se guarda en logs.jsonl para auditoría.

*Exportación de Reporte PDF*
Generado por la función create_pdf(), el archivo incluye:

- Encabezado con logo o avatar
- Resultado del modelo (riesgo y score)
- Plan de acción inicial
- Disclaimer ético
Ruta de descarga:
Reporte_NexusByte_YYYY-MM-DD.pdf

*Guardrails Éticos y de Fairness*

- El sistema no entrega diagnósticos médicos.
- Las respuestas incluyen siempre un disclaimer visible.
- Derivación automática a profesional si prediction == 1 (alto riesgo).
- Lenguaje inclusivo y empático.
- Validación de fairness en sexo y edad.

*Ejemplo de Cálculo de Riesgo*
{
  "feat_age": 45,
  "feat_sex": 0,
  "feat_imc": 27.5,
  "feat_whtr": 0.52,
  "feat_is_smoker": 0,
  "feat_sleep_hours": 7,
  "feat_activity_days": 3
}
Predicción: 0 → Bajo riesgo
Score: 0.18
Recomendación: Mantén actividad física regular y controla perímetro abdominal.

*Archivos de Log*
- app/results.csv
Registro histórico de predicciones:
| feat_age | feat_imc | feat_whtr | risk_score | prediction | timestamp |
| -------- | -------- | --------- | ---------- | ---------- | --------- |
-app/logs.jsonl

Bitácora del chat:
Dependencias Principales (requirements.txt)
streamlit==1.39.0
fastapi==0.115.0
uvicorn==0.30.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.0
requests==2.32.3
reportlab==4.2.0

*Reporte Técnico (Entregable 4)*
Documento complementario (PDF, 2–3 páginas) que describe:
1.Dataset y preprocesamiento
2.Ingeniería de features
3.Validación y fairness
4.Guardrails éticos
5.Limitaciones y próximos pasos

*Evaluación (según rúbrica Duoc UC)*
| Criterio | Descripción                       | Cumplimiento                      |
| -------- | --------------------------------- | --------------------------------- |
| A1–A3    | Modelo ML entrenado y validado    | listo                             |
| B1–B3    | Chat RAG funcional con guardrails | listo                             |
| C1–C2    | UI Streamlit con PDF descargable  | listo                             |
| D1       | Logs y scripts reproducibles      | listo                             |
| D2       | Documentación completa            | listo                             |
| E1–E3    | Presentación de resultados        | listo (pitch con demo y métricas) |


*Autores*
Francisca Sanhueza - Marcelo Veliz - Domingo Veloso - Yanina Silva
Proyecto “NexusByte” — Desafío Salud NHANES 2025
- Desarrollado para la asignatura BDY1102 – Base de Datos Aplicada II (Duoc UC)
