import os
import cv2
import face_recognition
import pickle
import numpy as np

# Configuración
OPERATORS_DIR = "server/operator-photo"  # Carpeta donde están las fotos 
OUTPUT_FILE = "operators/encodings.pkl"  # Donde guardaremos los datos procesados

def process_operator_photos():
    """Procesa las fotos de operadores pre-cargadas manualmente"""
    print("Procesando fotos de operadores...")
    
    # Verificar si existe la carpeta de operadores
    if not os.path.exists(OPERATORS_DIR):
        print(f"Error: La carpeta {OPERATORS_DIR} no existe")
        return False
    
    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Inicializar listas para guardar datos
    encodings = []
    names = []
    ids = []
    operators = {}
    
    # Recorrer las carpetas de operadores
    for operator_id in os.listdir(OPERATORS_DIR):
        operator_dir = os.path.join(OPERATORS_DIR, operator_id)
        
        # Verificar que sea un directorio
        if not os.path.isdir(operator_dir):
            continue
        
        # Leer archivo de información
        info_file = os.path.join(operator_dir, "info.txt")
        if not os.path.exists(info_file):
            print(f"Advertencia: No se encontró info.txt en {operator_dir}")
            continue
        
        # Leer nombre del operador
        with open(info_file, 'r', encoding='utf-8') as f:
            name = f.readline().strip()
        
        if not name:
            print(f"Advertencia: Archivo info.txt vacío en {operator_dir}")
            continue
        
        print(f"Procesando operador: {name} (ID: {operator_id})")
        
        # Registrar información del operador
        operators[operator_id] = {
            'id': operator_id,
            'name': name
        }
        
        # Procesar cada foto en la carpeta
        photos_processed = 0
        
        for file_name in os.listdir(operator_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(operator_dir, file_name)
                
                try:
                    # Cargar imagen
                    print(f"  Procesando imagen: {file_name}")
                    image = face_recognition.load_image_file(image_path)
                    
                    # Detectar rostros
                    face_locations = face_recognition.face_locations(image)
                    
                    if not face_locations:
                        print(f"  Advertencia: No se detectaron rostros en {file_name}")
                        continue
                    
                    # Si hay múltiples rostros, usar solo el primero
                    if len(face_locations) > 1:
                        print(f"  Advertencia: Se detectaron múltiples rostros en {file_name}, usando el primero")
                    
                    # Extraer encoding facial
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    
                    # Guardar datos
                    encodings.append(face_encoding)
                    names.append(name)
                    ids.append(operator_id)
                    
                    photos_processed += 1
                
                except Exception as e:
                    print(f"  Error al procesar {file_name}: {str(e)}")
        
        print(f"  {photos_processed} fotos procesadas para {name}")
    
    # Guardar encodings en archivo
    if encodings:
        data = {
            'encodings': encodings,
            'names': names,
            'ids': ids,
            'operators': operators
        }
        
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Datos guardados en {OUTPUT_FILE}")
        print(f"Total: {len(operators)} operadores, {len(encodings)} imágenes procesadas")
        return True
    else:
        print("No se procesaron imágenes")
        return False

if __name__ == "__main__":
    process_operator_photos()