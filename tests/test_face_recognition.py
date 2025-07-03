import cv2
import os
import face_recognition
import pickle
import numpy as np
import time

# Configuración
OPERATORS_DIR = "operators"

def load_known_faces():
    """Carga los rostros conocidos desde el archivo de encodings"""
    encodings_file = os.path.join(OPERATORS_DIR, "encodings.pkl")
    
    if not os.path.exists(encodings_file):
        print("No se encontró archivo de encodings. Primero debe registrar operadores.")
        return None, None, None, None
    
    print("Cargando rostros conocidos...")
    with open(encodings_file, 'rb') as f:
        data = pickle.load(f)
        return data['encodings'], data['names'], data['ids'], data['operators']

def main():
    """Función principal para probar el reconocimiento facial"""
    # Cargar rostros conocidos
    encodings, names, ids, operators = load_known_faces()
    
    if encodings is None:
        return
    
    print(f"Operadores cargados: {len(operators)}")
    for op_id, op_data in operators.items():
        print(f"  - {op_data['name']} (ID: {op_id})")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    print("Prueba de reconocimiento facial iniciada. Presione 'q' para salir.")
    
    # Para medir FPS
    prev_time = 0
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Redimensionar frame para procesamiento más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convertir de BGR a RGB
        
        # Localizar rostros en el frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Información a mostrar en la parte superior
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Para cada rostro encontrado
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Ajustar coordenadas al tamaño original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Comparar con rostros conocidos
            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)
            
            name = "Desconocido"
            face_distance = face_recognition.face_distance(encodings, face_encoding)
            
            # Si hay coincidencia
            if True in matches:
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = names[best_match_index]
                    operator_id = ids[best_match_index]
                    confidence = 1 - face_distance[best_match_index]
                    
                    # Información adicional del operador
                    operator_info = f"ID: {operator_id}"
                    
                    # Color verde para operadores autorizados
                    color = (0, 255, 0)
                else:
                    confidence = 0
                    operator_info = "No autorizado"
                    color = (0, 0, 255)  # Rojo para no autorizados
            else:
                confidence = 0
                operator_info = "No autorizado"
                color = (0, 0, 255)  # Rojo para no autorizados
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Añadir etiqueta debajo del rostro
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, operator_info, (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (left, bottom + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Mostrar frame
        cv2.imshow("Reconocimiento Facial", frame)