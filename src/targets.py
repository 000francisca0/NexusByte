# Contenido para: src/targets.py (VERSIÓN 3 - Final)

import pandas as pd
import numpy as np
import os

# --- Configuración ---
DATA_DIR = "data/processed"
INPUT_TRAIN = os.path.join(DATA_DIR, "train_dataset.csv")
INPUT_TEST = os.path.join(DATA_DIR, "test_dataset.csv")

OUTPUT_TRAIN = os.path.join(DATA_DIR, "train_with_target.csv")
OUTPUT_TEST = os.path.join(DATA_DIR, "test_with_target.csv")

# Variables de laboratorio (SOLO EXISTEN EN TRAIN)
SYSTOLIC_VAR = 'BPXSY2'
DIASTOLIC_VAR = 'BPXDI2'
TARGET_NAME = 'TARGET_HIPERTENSION'

# Umbral de riesgo: (Sistólica >= 130) O (Diastólica >= 80)
SYSTOLIC_THRESHOLD = 130
DIASTOLIC_THRESHOLD = 80

def create_target_variable(df, target_name):
    """
    Crea la columna objetivo binaria basada en los umbrales de presión arterial.
    Esta función asume que las columnas (BPXSY2, BPXDI2) existen.
    """
    
    # 1. Verificar que las columnas de laboratorio existan
    if not all(col in df.columns for col in [SYSTOLIC_VAR, DIASTOLIC_VAR]):
        print(f"  ERROR: Columnas '{SYSTOLIC_VAR}' o '{DIASTOLIC_VAR}' no encontradas.")
        return None
    
    # 2. Crear el target
    df[target_name] = 0
    condicion_hipertension = (
        (df[SYSTOLIC_VAR] >= SYSTOLIC_THRESHOLD) | 
        (df[DIASTOLIC_VAR] >= DIASTOLIC_THRESHOLD)
    )
    df.loc[condicion_hipertension, target_name] = 1
    
    # 3. Manejar valores nulos (filas donde no se pudo medir la presión)
    initial_rows = len(df)
    df = df.dropna(subset=[SYSTOLIC_VAR, DIASTOLIC_VAR])
    rows_dropped = initial_rows - len(df)
    
    if rows_dropped > 0:
        print(f"  Aviso: Se eliminaron {rows_dropped} filas por tener '{SYSTOLIC_VAR}' o '{DIASTOLIC_VAR}' nulo.")
    
    df[target_name] = df[target_name].astype(int)
    
    return df

def main():
    print(f"--- Iniciando script: src/targets.py (Target: Hipertensión) ---")
    
    # --- Procesar datos de ENTRENAMIENTO ---
    print(f"\nProcesando {INPUT_TRAIN}...")
    try:
        train_df = pd.read_csv(INPUT_TRAIN)
        train_df_with_target = create_target_variable(train_df, TARGET_NAME)
        
        if train_df_with_target is not None:
            train_df_with_target.to_csv(OUTPUT_TRAIN, index=False)
            print(f"✅ Archivo de entrenamiento guardado en: {OUTPUT_TRAIN}")
            print(f"   Distribución del target: \n{train_df_with_target[TARGET_NAME].value_counts(normalize=True)}")
            
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {INPUT_TRAIN}")
        return

    # --- Procesar datos de PRUEBA ---
    # Esto es un set de prueba ciego. No podemos calcular el target.
    # Solo crearemos la columna 'TARGET_HIPERTENSION' como un placeholder (NaN).
    print(f"\nProcesando {INPUT_TEST} (Set de prueba ciego)...")
    try:
        test_df = pd.read_csv(INPUT_TEST)
        
        # Verificar si las columnas BPX existen (no deberían)
        if SYSTOLIC_VAR in test_df.columns:
            print("  Aviso: Se encontró columna de target en el test set. Esto es inusual.")
            # Si existieran por error, las procesamos.
            test_df_with_target = create_target_variable(test_df, TARGET_NAME)
        else:
            # Este es el camino esperado: crear un placeholder
            print(f"  No se encontraron columnas de target (ej: {SYSTOLIC_VAR}). Es un set ciego.")
            print(f"  Creando columna '{TARGET_NAME}' con valores nulos (NaN).")
            test_df[TARGET_NAME] = np.nan
            test_df_with_target = test_df
        
        test_df_with_target.to_csv(OUTPUT_TEST, index=False)
        print(f"✅ Archivo de prueba (ciego) guardado en: {OUTPUT_TEST}")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {INPUT_TEST}")
        return
        
    print("\n--- Proceso de creación de targets completado ---")

if __name__ == "__main__":
    main()