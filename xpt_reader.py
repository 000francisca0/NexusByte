import xport.v56
import requests
import pandas as pd
import os
from urls_and_columns import (
    TRAIN_URLS,
    TEST_URLS,
    COLUMNS_TO_SAVE
)

# --- Tus funciones de lectura (sin cambios) ---
def request_and_save_xpt_data_from_url(url):
    """
    Descarga y guarda los datos XPT de la URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open('temp.xpt', 'wb') as temp_file:
        temp_file.write(response.content)

def read_xpt_file(file_path):
    """
    Lee un archivo XPT local.
    """
    with open(file_path, 'rb') as f:
        library = xport.v56.load(f)
    return library

# --- Función de procesamiento (CON LÓGICA DE SUEÑO MEJORADA) ---
def make_csv_from_url(years_and_urls, columns_to_save, output_csv):
    """
    Procesa un conjunto de URLs (Train o Test) y las guarda en un CSV.
    """
    df = pd.DataFrame()
    for year, url in years_and_urls.items():
        print(f"Procesando {output_csv}, Año: {year}...")
        try:
            request_and_save_xpt_data_from_url(url)
            xpt_data = read_xpt_file('temp.xpt')
        except Exception as e:
            print(f"  ERROR: No se pudo descargar o leer {url}. Error: {e}")
            continue

        for table_name, table_data in xpt_data.items():
            table_df = pd.DataFrame(table_data)
            table_df["year"] = year
            
            # --- Lógica Mejorada para Columnas ---
            
            # Manejar el caso especial de sueño (SLD010H/SLD012)
            # Los unifica en una nueva columna 'SLD_HOURS'
            if 'SLD010H' in table_df.columns and 'SLD012' in table_df.columns:
                table_df['SLD_HOURS'] = table_df['SLD010H'].fillna(table_df['SLD012'])
            elif 'SLD010H' in table_df.columns:
                table_df['SLD_HOURS'] = table_df['SLD010H']
            elif 'SLD012' in table_df.columns:
                table_df['SLD_HOURS'] = table_df['SLD012']
            
            # Ahora, actualizamos la lista de columnas a guardar
            final_cols_to_save = []
            
            # Usamos startswith() para chequear el nombre descriptivo
            if output_csv.startswith("SleepDisorder"): 
                for col in columns_to_save:
                    if col not in ['SLD010H', 'SLD012']:
                        final_cols_to_save.append(col)
                final_cols_to_save.append('SLD_HOURS') # Añade la nueva columna unificada
            else:
                final_cols_to_save = columns_to_save

            # Seleccionar solo las columnas que existen en este archivo
            cols_to_use = [col for col in final_cols_to_save if col in table_df.columns]
            
            if "SEQN" not in cols_to_use:
                cols_to_use.insert(0, "SEQN")
            if "year" not in table_df.columns:
                table_df["year"] = year
            if "year" not in cols_to_use:
                cols_to_use.append("year")
            
            # --- Fin de Lógica Mejorada ---
            
            final_cols_exist = [col for col in cols_to_use if col in table_df.columns]
            
            filtered_df = table_df[final_cols_exist]
            df = pd.concat([df, filtered_df], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"✅ Archivo guardado: {output_csv}")


# --- Bloque Principal (MODIFICADO CON NUEVOS NOMBRES) ---
    
# --- MODIFICACIÓN: Mapeo de nombres de archivos ---
FILENAME_MAP = {
    "BPX": "BloodPressure",
    "DEMO": "AgeAndSex",
    "BMX": "BodyMeasures",
    "PAQ": "PhysicalActivity",
    "SLQ": "SleepDisorder",
    "DBQ": "Dietary",
    "BPQ": "Hypertension", # Cuestionario de HTA (Autodeclarada)
    "SMQ": "TabacoUse"
}

if __name__ == "__main__":
    
    # 1. Procesar y guardar archivos de ENTRENAMIENTO (2007-2016)
    print("--- INICIANDO DESCARGA DE DATOS DE ENTRENAMIENTO ---")
    for file_key, descriptive_name in FILENAME_MAP.items():
        make_csv_from_url(
            years_and_urls=TRAIN_URLS[file_key],
            columns_to_save=COLUMNS_TO_SAVE[file_key],
            output_csv=f"{descriptive_name}_TRAIN.csv" # ¡Nombre cambiado!
        )

    # 2. Procesar y guardar archivos de TEST (2017-2020)
    print("\n--- INICIANDO DESCARGA DE DATOS DE TEST ---")
    for file_key, descriptive_name in FILENAME_MAP.items():
        make_csv_from_url(
            years_and_urls=TEST_URLS[file_key],
            columns_to_save=COLUMNS_TO_SAVE[file_key],
            output_csv=f"{descriptive_name}_TEST.csv" # ¡Nombre cambiado!
        )

    # 3. Limpiar archivo temporal
    try:
        os.remove('temp.xpt')
    except OSError:
        pass

    print("\n--- PROCESO DE CARGA DE DATOS COMPLETADO ---")