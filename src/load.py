#merge 
 
import pandas as pd
import glob
import os


# Define los directorios
# Asumimos que este script se ejecuta desde el directorio raíz (NexusByte/)
# así: python src/load.py
ROOT_DIR = "." 
OUTPUT_DIR = "data/processed"

# Asegurarse de que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_files(file_suffix, output_filename):
    """
    Encuentra todos los CSV con un sufijo, los une por 'SEQN' y los guarda.
    """
    print(f"\n--- Iniciando merge para: {output_filename} ---")
    
    # 1. Encontrar todos los archivos CSV que coincidan
    files_to_merge = glob.glob(f"{ROOT_DIR}/*{file_suffix}")
    
    if not files_to_merge:
        print(f"AVISO: No se encontraron archivos con el sufijo {file_suffix}")
        return

    print(f"Se encontraron {len(files_to_merge)} archivos para unir.")
    
    # 2. Cargar el primer archivo como base
    # (Usaremos AgeAndSex como base, ya que es el archivo demográfico)
    base_file = f"{ROOT_DIR}/AgeAndSex{file_suffix}"
    if not os.path.exists(base_file):
        print(f"ERROR: No se encuentra el archivo base {base_file}.")
        return
        
    base_df = pd.read_csv(base_file)
    print(f"  Cargando base: {base_file} (Filas: {len(base_df)})")
    
    # Lista de archivos restantes para unir
    files_to_merge.remove(base_file)

    # 3. Iterar y unir (merge) los archivos restantes
    for file_path in files_to_merge:
        df_to_merge = pd.read_csv(file_path)
        print(f"  Uniendo con: {file_path} (Filas: {len(df_to_merge)})")
        
        # Eliminar la columna 'year' de los archivos secundarios para evitar
        # conflictos (year_x, year_y)
        if 'year' in df_to_merge.columns:
            df_to_merge = df_to_merge.drop(columns=['year'])

        # Unir usando 'SEQN' (ID del participante)
        # 'how=outer' asegura que no perdamos participantes si faltan en un archivo
        base_df = pd.merge(base_df, df_to_merge, on='SEQN', how='outer')

    # 4. Guardar el archivo final
    final_output_path = os.path.join(OUTPUT_DIR, output_filename)
    base_df.to_csv(final_output_path, index=False)
    
    print(f"\n✅ Merge completado. Archivo guardado en: {final_output_path}")
    print(f"  Dimensiones finales (Filas, Columnas): {base_df.shape}")

if __name__ == "__main__":
    
    # Crear la base de datos de ENTRENAMIENTO
    merge_files(
        file_suffix="_TRAIN.csv",
        output_filename="train_dataset.csv"
    )
    
    # Crear la base de datos de TEST
    merge_files(
        file_suffix="_TEST.csv",
        output_filename="test_dataset.csv"
    )
    
    print("\n--- PROCESO DE MERGE COMPLETADO ---")