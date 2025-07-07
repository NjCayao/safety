import os
import face_recognition
import pickle
import time
import glob
from datetime import datetime

# Obtener el directorio donde está este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def update_progress(progress):
    """Actualiza el archivo de progreso con el porcentaje actual"""
    # Crear en la carpeta del script
    progress_file = os.path.join(SCRIPT_DIR, "update_progress.txt")
    with open(progress_file, "w") as f:
        f.write(str(progress))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    photos_dir = os.path.join(base_dir, "server", "operator-photo")
    
    print(f"Directorio base: {base_dir}")
    print(f"Buscando fotos en: {photos_dir}")
    
    # Inicializa el archivo de progreso
    update_progress(0)

    # Archivo de log en la carpeta del script
    log_file = os.path.join(SCRIPT_DIR, "update_log.txt")
    
    # Verifica si puede acceder al directorio de fotos
    if not os.path.exists(photos_dir):
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No se puede acceder al directorio de fotos: {photos_dir}\n")
        update_progress(100)  # Marca como completado para evitar que la interfaz se quede esperando
        return
    
    # Encuentra todas las carpetas de operadores (por DNI)
    operator_folders = [f for f in os.listdir(photos_dir) if os.path.isdir(os.path.join(photos_dir, f))]
    total_operators = len(operator_folders)
    
    if total_operators == 0:
        update_progress(100)
        return
    
    # Diccionario para almacenar los encodings
    encodings = []
    names = []
    ids = []
    operators = {}
    
    # Procesa cada carpeta de operador
    for i, operator_dni in enumerate(operator_folders):
        operator_path = os.path.join(photos_dir, operator_dni)
        
        # Intenta leer el archivo info.txt para obtener el nombre del operador
        operator_name = operator_dni  # Valor predeterminado
        info_file = os.path.join(operator_path, "info.txt")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r") as f:
                    info_content = f.read().strip()
                    # Asume que la primera línea es el nombre
                    operator_name = info_content.split('\n')[0]
            except Exception:
                pass
        
        # Busca todas las imágenes en la carpeta del operador
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(operator_path, ext)))
        
        # Si no hay imágenes, continúa con el siguiente operador
        if not image_files:
            continue
        
        # Procesa cada imagen del operador
        operator_encodings = []
        for img_path in image_files:
            try:
                # Carga la imagen
                image = face_recognition.load_image_file(img_path)
                
                # Detecta las caras en la imagen
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    # Genera encoding para la primera cara encontrada
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings.append(encoding)
                    names.append(operator_name)
                    ids.append(operator_dni)
                    
                    # También guarda el encoding para este operador
                    operator_encodings.append(encoding)
            except Exception:
                pass
        
        # Si se encontraron encodings para este operador, guárdalo en el diccionario de operadores
        if operator_encodings:
            operators[operator_dni] = {
                "id": operator_dni,
                "name": operator_name
            }
        
        # Actualiza el progreso (i+1 porque i comienza en 0)
        progress = int(((i + 1) / total_operators) * 100)
        update_progress(progress)
        time.sleep(0.1)  # Pequeña pausa para simular procesamiento
    
    # Estructura de datos final
    data = {
        "encodings": encodings,
        "names": names,
        "ids": ids,
        "operators": operators
    }
    
    # Guarda todos los encodings en el archivo pickle
    output_file = os.path.join(SCRIPT_DIR, "encodings.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    # Asegura que el progreso final sea 100%
    update_progress(100)

     # Log final
    with open(log_file, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Actualización completada. {len(encodings)} encodings guardados\n")
    
    # Asegura que el progreso final sea 100%
    update_progress(100)
    print(f"✅ Proceso completado. {len(encodings)} encodings guardados")

if __name__ == "__main__":
    main()