# Contenido para: src/model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import os
import joblib # Se usará para guardar el modelo

# --- Configuración ---
DATA_DIR = "data/processed"
MODEL_DIR = "models"
INPUT_TRAIN = os.path.join(DATA_DIR, "train_final_features.csv")

TARGET_COL = "TARGET_HIPERTENSION"

# Asegurarse de que el directorio de modelos exista
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "hypertension_model.joblib")

def train_model():
    print(f"--- Iniciando script: src/model.py ---")
    
    # 1. Cargar datos
    try:
        df_train = pd.read_csv(INPUT_TRAIN)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {INPUT_TRAIN}")
        print("Asegúrate de haber corrido src/features.py primero.")
        return

    print(f"Datos cargados. Dimensiones: {df_train.shape}")
    
    # 2. Separar Features (X) y Target (y)
    feature_cols = [col for col in df_train.columns if col.startswith('feat_')]
    
    if not feature_cols:
        print("ERROR: No se encontraron 'features' (columnas 'feat_...').")
        return
        
    X = df_train[feature_cols]
    y = df_train[TARGET_COL]
    
    print(f"Separando {len(feature_cols)} features y el target '{TARGET_COL}'.")
    
    # 3. Dividir en set de Entrenamiento y Validación
    # Usaremos el 20% de los datos de entrenamiento para validar el modelo
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2,    # 20% para validación
        random_state=42,  # Para resultados reproducibles
        stratify=y        # Asegura que la proporción de 0s y 1s sea igual en ambos sets
    )
    
    print(f"  Datos de entrenamiento: {X_train.shape}")
    print(f"  Datos de validación: {X_val.shape}")
    
    # 4. Entrenar el modelo (XGBoost)
    print("\nIniciando entrenamiento de XGBoost...")
    
    # El 'scale_pos_weight' es crucial porque tus datos están desbalanceados
    # (vimos que ~69% eran 0 y ~31% eran 1)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        n_estimators=100,
        early_stopping_rounds=10, # Parar si no mejora en 10 rondas
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], # Evaluar contra el set de validación
        verbose=False # Puedes ponerlo en True si quieres ver el progreso
    )
    
    print("✅ Entrenamiento completado.")
    
    # 5. Evaluar el modelo
    print("\n--- Evaluación del Modelo (en set de validación) ---")
    preds = model.predict(X_val)
    proba_preds = model.predict_proba(X_val)[:, 1] # Probabilidades para la clase 1

    # Métricas del PDF
    auroc = roc_auc_score(y_val, proba_preds)
    auprc = average_precision_score(y_val, proba_preds)
    
    print(f"  AUROC (Area Under ROC Curve): {auroc:.4f}")
    print(f"  AUPRC (Area Under PR Curve):  {auprc:.4f}")
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, preds, target_names=['No Hipertenso (0)', 'Hipertenso (1)']))
    
    # 6. Guardar el modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Modelo guardado exitosamente en: {MODEL_PATH}")
    
    print("\n--- Proceso de entrenamiento completado ---")

if __name__ == "__main__":
    train_model()