# Contenido para: src/features.py (VERSIÓN 2 - Hipertensión)

import pandas as pd
import os

# --- Configuración ---
DATA_DIR = "data/processed"
INPUT_TRAIN = os.path.join(DATA_DIR, "train_with_target.csv")
INPUT_TEST = os.path.join(DATA_DIR, "test_with_target.csv")

OUTPUT_TRAIN = os.path.join(DATA_DIR, "train_final_features.csv")
OUTPUT_TEST = os.path.join(DATA_DIR, "test_final_features.csv")

# 1. REGLA ANTI-FUGA: Lista de variables a ELIMINAR
# Ya que predecimos HIPERTENSIÓN, no podemos usar ninguna
# medición de presión arterial como feature.
LEAKY_VARS = [
    'BPXSY1', 'BPXDI1', # Presión Arterial - Medición 1
    'BPXSY2', 'BPXDI2', # Presión Arterial - Medición 2 (Usada para el target)
    'BPXSY3', 'BPXDI3', # Presión Arterial - Medición 3
    'BPXSY4', 'BPXDI4', # Presión Arterial - Medición 4 (si existe)
    'BPQ020', # Alguna vez le dijeron que tenía presión alta
    'BPQ030', # Está tomando medicamentos para presión
]

# 2. VARIABLES DE ENTRADA (Features)
# Mapeo de nombres "reales" (de tu head) a nombres genéricos
COLS_RAW = {
    'age': 'RIDAGEYR',
    'sex': 'RIAGENDR',        # 1=Hombre, 2=Mujer
    'height': 'BMXHT',      # en cm
    'weight': 'BMXWT',      # en kg
    'waist': 'BMXWAIST',    # Circunferencia de cintura en cm
    'sleep_hours': 'SLD_HOURS', # Horas de sueño (¡Detectado de tu head!)
    'is_smoker': 'SMQ020',    # 1=Sí, 2=No
    'activity_days': 'PAQ650', # Días de actividad física moderada
}

# 3. Columnas a mantener al final
ID_COL = 'SEQN'
TARGET_COL = 'TARGET_HIPERTENSION' # <-- ¡Actualizado!


def engineer_features(df):
    """
    Crea nuevas features y limpia las existentes.
    """
    df_feat = pd.DataFrame()
    
    # Verificar que las columnas base existan
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        print(f"ERROR: No se encontró {ID_COL} o {TARGET_COL} en {DATA_DIR}/train_with_target.csv")
        return None

    # Mantener ID y Target
    df_feat[ID_COL] = df[ID_COL]
    df_feat[TARGET_COL] = df[TARGET_COL]

    # --- Ingeniería de Features ---
    
    # 1. IMC (Índice de Masa Corporal)
    if COLS_RAW['weight'] in df.columns and COLS_RAW['height'] in df.columns:
        df_feat['feat_imc'] = df[COLS_RAW['weight']] / (df[COLS_RAW['height']] / 100) ** 2
    
    # 2. Relación Cintura-Altura (WHtR - Waist-to-Height Ratio)
    if COLS_RAW['waist'] in df.columns and COLS_RAW['height'] in df.columns:
        df_feat['feat_whtr'] = df[COLS_RAW['waist']] / df[COLS_RAW['height']]

    # --- Limpieza y Renombrado ---
    
    # 3. Edad (ya está lista)
    if COLS_RAW['age'] in df.columns:
        df_feat['feat_age'] = df[COLS_RAW['age']]
    
    # 4. Sexo (Mapear a 0 y 1)
    if COLS_RAW['sex'] in df.columns:
        df_feat['feat_sex'] = df[COLS_RAW['sex']].map({1: 0, 2: 1}) # 1=Hombre -> 0, 2=Mujer -> 1
    
    # 5. Tabaquismo (Mapear a 0 y 1)
    if COLS_RAW['is_smoker'] in df.columns:
        df_feat['feat_is_smoker'] = df[COLS_RAW['is_smoker']].map({1: 1, 2: 0}) # 1=Sí -> 1, 2=No
    else:
        print("  Aviso: 'SMQ020' (tabaquismo) no encontrado. Se omitirá.")

    # 6. Horas de sueño
    if COLS_RAW['sleep_hours'] in df.columns:
        df_feat['feat_sleep_hours'] = df[COLS_RAW['sleep_hours']].replace([77, 99], pd.NA)
    else:
        print("  Aviso: 'SLD_HOURS' (sueño) no encontrado. Se omitirá.")
        
    # 7. Días de actividad física
    if COLS_RAW['activity_days'] in df.columns:
        df_feat['feat_activity_days'] = df[COLS_RAW['activity_days']].replace([7, 9], pd.NA)
    else:
        print("  Aviso: 'PAQ650' (actividad) no encontrado. Se omitirá.")

    return df_feat


def main():
    print(f"--- Iniciando script: src/features.py (Target: Hipertensión) ---")
    
    try:
        train_df = pd.read_csv(INPUT_TRAIN)
        test_df = pd.read_csv(INPUT_TEST)
    except FileNotFoundError:
        print(f"ERROR: No se encontraron los archivos de entrada (ej: {INPUT_TRAIN})")
        print("Por favor, ejecuta 'python src/targets.py' primero.")
        return

    # --- 1. Aplicar Regla Anti-Fuga ---
    print(f"Eliminando variables de fuga (anti-leakage rule)...")
    cols_to_drop = [col for col in LEAKY_VARS if col in train_df.columns]
    print(f"  Columnas eliminadas: {cols_to_drop}")
    
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

    # --- 2. Ingeniería de Features ---
    print("Iniciando ingeniería de features...")
    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)
    
    if train_feat is None or test_feat is None:
        print("Hubo un error en engineer_features. Abortando.")
        return
        
    print("  Features creadas: 'feat_imc', 'feat_whtr', etc.")

    # --- 3. Imputación (Manejo de Nulos) ---
    feature_cols = [col for col in train_feat.columns if col.startswith('feat_')]
    print(f"Imputando valores nulos para {len(feature_cols)} features...")
    
    imputer = train_feat[feature_cols].median()
    
    train_feat[feature_cols] = train_feat[feature_cols].fillna(imputer)
    test_feat[feature_cols] = test_feat[feature_cols].fillna(imputer)
    
    print("  Imputación completada usando la mediana de 'train'.")
    
    # --- 4. Guardar Archivos Finales ---
    train_feat.to_csv(OUTPUT_TRAIN, index=False)
    test_feat.to_csv(OUTPUT_TEST, index=False)
    
    print(f"\n✅ Archivo final de entrenamiento guardado en: {OUTPUT_TRAIN}")
    print(f"   Dimensiones: {train_feat.shape}")
    print(f"\n✅ Archivo final de prueba guardado en: {OUTPUT_TEST}")
    print(f"   Dimensiones: {test_feat.shape}")
    
    print("\n--- Proceso de creación de features completado ---")

if __name__ == "__main__":
    main()