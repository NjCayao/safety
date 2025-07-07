import cv2
import os
import numpy as np
import time
import face_recognition
import pickle
import argparse

# Configuración
OPERATORS_DIR = "operators"
FACES_PER_OPERATOR = 4  # Número de fotos a capturar

def create_operator_directory(operator_id, name):
    """Crea los directorios para almacenar la información del operador"""
    # Directorio principal del operador
    operator_dir = os.path.join(OPERATORS_DIR, operator_id)
    os.makedirs(operator_dir, exist_ok=True)
    
    # Directorio para las imágenes faciales
    faces_dir = os.path.join(operator_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    
    # Guardar información del operador
    with open(os.path.join(operator_dir, "info.txt"), 'w') as f:
        f.write(f"{name}\n")
        f.write(f"Fecha de registro: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return faces_dir

def save_encodings(encodings, names, ids, operators):
    """Guarda los encodings faciales para su uso posterior"""
    # Crear directorio si no existe
    os.makedirs(OPERATORS_DIR, exist_ok=True)
    
    # Guardar encodings
    data = {
        'encodings': encodings,
        'names': names,
        'ids': ids,
        'operators': operators
    }
    
    with open(os.path.join(OPERATORS_DIR, "encodings.pkl"), 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Encodings guardados para {len(names)} imágenes")

def load_existing_encodings():
    """Carga los encodings existentes si existen"""
    encodings_file = os.path.join(OPERATORS_DIR, "encodings.pkl")
    
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names'], data['ids'], data['assets/operators']
    
    return [], [], [], {}

def register_new_operator(name, operator_id=None):
    """Registra un nuevo operador capturando imágenes de su rostro"""
    if operator_id is None:
        operator_id = f"op_{int(time.time())}"
    
    print(f"Registrando operador: {name} (ID: {operator_id})")
    print(f"Se tomarán {FACES_PER_OPERATOR} fotos. Siga las instrucciones en pantalla.")
    
    # Cargar encodings existentes
    known_encodings, known_names, known_ids, operators = load_existing_encodings()
    
    # Crear directorios para el operador
    faces_dir = create_operator_directory(operator_id, name)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return False
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_count = 0
    total_faces = FACES_PER_OPERATOR
    
    # Bucle principal
    while face_count < total_faces:
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        # Mostrar instrucciones
        instruction = f"Foto {face_count+1}/{total_faces}"
        if face_count == 0:
            pose = "Mire directamente a la cámara"
        elif face_count == 1:
            pose = "Gire ligeramente a la izquierda"
        elif face_count == 2:
            pose = "Gire ligeramente a la derecha"
        else:
            pose = "Incline ligeramente la cabeza"
        
        # Mostrar marco para posicionar el rostro
        h, w = frame.shape[:2]
        margin_x, margin_y = w//4, h//4
        cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 255, 0), 2)
        
        # Mostrar instrucciones en pantalla
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, pose, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Presione ESPACIO para capturar", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Presione ESC para cancelar", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Registro de Operador", frame)
        
        # Capturar tecla
        key = cv2.waitKey(1) & 0xFF
        
        # ESC para salir
        if key == 27:
            print("Registro cancelado por el usuario")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        # ESPACIO para capturar foto
        if key == 32:  # Tecla ESPACIO
            # Detectar rostros en la imagen
            rgb_frame = frame[:, :, ::-1]  # Convertir de BGR a RGB
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                print("No se detectó ningún rostro. Intente de nuevo.")
                continue
            
            # Si hay múltiples rostros, usar solo el primero
            if len(face_locations) > 1:
                print("Se detectaron múltiples rostros. Por favor, asegúrese de que solo haya una persona en la imagen.")
                continue
            
            # Guardar imagen
            img_path = os.path.join(faces_dir, f"face_{face_count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Imagen {face_count+1} guardada")
            
            # Extraer encoding
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            
            # Añadir a listas
            known_encodings.append(face_encoding)
            known_names.append(name)
            known_ids.append(operator_id)
            
            face_count += 1
            
            # Mostrar confirmación
            confirm_frame = frame.copy()
            cv2.putText(confirm_frame, f"Foto {face_count}/{total_faces} capturada", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registro de Operador", confirm_frame)
            cv2.waitKey(1000)  # Mostrar confirmación por 1 segundo
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Actualizar operadores
    operators[operator_id] = {
        'id': operator_id,
        'name': name
    }
    
    # Guardar encodings
    save_encodings(known_encodings, known_names, known_ids, operators)
    
    print(f"Operador {name} registrado exitosamente con ID: {operator_id}")
    return True

def list_operators():
    """Lista todos los operadores registrados"""
    print("\n=== Operadores Registrados ===")
    
    encodings_file = os.path.join(OPERATORS_DIR, "encodings.pkl")
    if not os.path.exists(encodings_file):
        print("No hay operadores registrados")
        return
    
    with open(encodings_file, 'rb') as f:
        data = pickle.load(f)
        operators = data['assets/operators']
    
    if not operators:
        print("No hay operadores registrados")
        return
    
    print(f"Total de operadores: {len(operators)}")
    for op_id, op_data in operators.items():
        print(f"ID: {op_id}, Nombre: {op_data['name']}")

if __name__ == "__main__":
    # Crear directorio de operadores si no existe
    os.makedirs(OPERATORS_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Registro de operadores para el sistema de seguridad')
    parser.add_argument('--list', action='store_true', help='Listar operadores registrados')
    parser.add_argument('--name', type=str, help='Nombre del operador a registrar')
    parser.add_argument('--id', type=str, help='ID opcional para el operador')
    
    args = parser.parse_args()
    
    if args.list:
        list_operators()
    elif args.name:
        register_new_operator(args.name, args.id)
    else:
        # Modo interactivo
        print("=== Sistema de Registro de Operadores ===")
        print("1. Registrar nuevo operador")
        print("2. Listar operadores registrados")
        print("3. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            name = input("Nombre del operador: ")
            register_new_operator(name) 
        elif choice == "2":
            list_operators()
        else:
            print("Saliendo del sistema de registro")