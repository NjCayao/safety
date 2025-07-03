import cv2
import numpy as np
import os
import time
import pygame

# Configuración
MODEL_DIR = "models"
CONFIG_FILE = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3.weights")
CLASSES_FILE = os.path.join(MODEL_DIR, "coco.names")
AUDIO_DIR = "audio"

# Clases que queremos detectar (basadas en COCO dataset)
TARGET_CLASSES = {
    "cell phone": {"id": None, "color": (0, 0, 255)},  # Rojo
    "cigarette": {"id": None, "color": (0, 165, 255)}  # Naranja
}



# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.5

def play_alarm(alarm_file="alarma.mp3"):
    """Reproduce una alarma de audio"""
    try:
        # Inicializar pygame si no está inicializado
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Ruta al archivo de audio
        alarm_path = os.path.join(AUDIO_DIR, alarm_file)
        
        if os.path.exists(alarm_path):
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()
            print(f"Reproduciendo alarma: {alarm_path}")
        else:
            print(f"Archivo de audio no encontrado: {alarm_path}")
    except Exception as e:
        print(f"Error al reproducir alarma: {e}")

def load_yolo_model():
    """Carga el modelo YOLO para detección de objetos"""
    print("Cargando modelo YOLO...")
    
    # Verificar si existen los archivos
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: No se encontró el archivo de configuración: {CONFIG_FILE}")
        return None
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: No se encontró el archivo de pesos: {WEIGHTS_FILE}")
        return None
    
    if not os.path.exists(CLASSES_FILE):
        print(f"Error: No se encontró el archivo de clases: {CLASSES_FILE}")
        return None
    
    # Cargar nombres de clases
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Mapear IDs de clases objetivo
    for i, class_name in enumerate(classes):
        if class_name in TARGET_CLASSES:
            TARGET_CLASSES[class_name]["id"] = i
    
    # Si "cigarette" no está en el conjunto de datos COCO, buscaremos "bottle" o similares
    if TARGET_CLASSES["cigarette"]["id"] is None:
        for i, class_name in enumerate(classes):
            if class_name == "bottle":
                print("Nota: 'cigarette' no está en COCO, usando 'bottle' como sustituto")
                TARGET_CLASSES["cigarette"]["id"] = i
                break
    
    # Cargar la red neuronal
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    
    # Usar CPU o GPU si está disponible
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # Para usar GPU, descomentar la siguiente línea y comentar la anterior:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    print("Modelo YOLO cargado correctamente")
    return net, classes

def detect_objects(frame, net, classes, face_locations=None):
    """
    Detecta objetos en el frame usando YOLO
    
    Args:
        frame: Frame de video a analizar
        net: Modelo YOLO cargado
        classes: Lista de nombres de clases
        face_locations: Ubicaciones de rostros detectados (opcional)
    
    Returns:
        frame: Frame con anotaciones
        detections: Lista de objetos detectados
    """
    height, width = frame.shape[:2]
    
    # Crear blob desde la imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Pasar blob por la red
    net.setInput(blob)
    
    # Obtener capas de salida
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    
    # Inicializar listas
    boxes = []
    confidences = []
    class_ids = []
    
    # Procesar cada detección
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filtrar por confianza y clases objetivo
            target_ids = [info["id"] for info in TARGET_CLASSES.values() if info["id"] is not None]
            if confidence > CONFIDENCE_THRESHOLD and class_id in target_ids:
                # Calcular coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas de la esquina superior izquierda
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Aplicar non-maximum suppression para eliminar detecciones duplicadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
    
    # Lista de objetos detectados
    detections = []
    
    # Si hay rostros detectados, solo consideramos objetos cerca de los rostros
    face_proximity = False
    if face_locations and len(face_locations) > 0:
        face_proximity = True
    
    # Dibujar detecciones
    if len(boxes) > 0:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Obtener nombre de clase
                detected_class = classes[class_id]
                
                # Si es una clase objetivo
                for target_name, target_info in TARGET_CLASSES.items():
                    if target_info["id"] == class_id:
                        # Verificar proximidad al rostro si hay rostros detectados
                        if face_proximity:
                            near_face = False
                            for face_top, face_right, face_bottom, face_left in face_locations:
                                # Calcular centro del objeto y del rostro
                                obj_center_x = x + w/2
                                obj_center_y = y + h/2
                                face_center_x = (face_left + face_right)/2
                                face_center_y = (face_top + face_bottom)/2
                                
                                # Calcular distancia entre centros
                                distance = np.sqrt((obj_center_x - face_center_x)**2 + 
                                                 (obj_center_y - face_center_y)**2)
                                
                                # Calcular diagonal del rostro para normalizar
                                face_diag = np.sqrt((face_right - face_left)**2 + 
                                                  (face_bottom - face_top)**2)
                                
                                # Si el objeto está cerca del rostro (ajustar umbral según necesidad)
                                if distance < face_diag * 2:
                                    near_face = True
                                    break
                            
                            if not near_face:
                                continue
                        
                        # Dibujar rectángulo
                        color = target_info["color"]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Añadir etiqueta
                        label = f"{target_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Añadir a la lista de detecciones
                        detections.append((target_name, confidence))
                        break
    
    return frame, detections

def main():
    """Función principal para la detección de comportamientos peligrosos"""
    print("Iniciando sistema de detección de comportamientos peligrosos...")
    
    # Crear directorios si no existen
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Verificar si los archivos del modelo existen
    if not (os.path.exists(CONFIG_FILE) and os.path.exists(WEIGHTS_FILE) and os.path.exists(CLASSES_FILE)):
        print("Archivos del modelo YOLO no encontrados. Por favor, descarga los archivos necesarios:")
        print(f"1. Archivo de configuración: {CONFIG_FILE}")
        print(f"2. Archivo de pesos: {WEIGHTS_FILE}")
        print(f"3. Archivo de clases: {CLASSES_FILE}")
        print("Consulta la documentación para más detalles.")
        return
    
    # Inicializar pygame para reproducción de audio
    pygame.init()
    
    # Cargar modelo YOLO
    model_data = load_yolo_model()
    if model_data is None:
        return
    
    net, classes = model_data
    
    # Cargar detector facial para correlacionar objetos con la posición de la cara
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
        
    # Crear ventana redimensionable con tamaño específico
    cv2.namedWindow("Detección de Comportamientos Peligrosos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Comportamientos Peligrosos", 800, 600)
    
    print("Sistema de detección iniciado. Presione 'q' para salir.")
    
    
    # Para calcular FPS
    prev_time = 0
    
    # Para evitar alertas repetitivas
    last_detection_time = {}
    cooldown_period = 3  # Segundos
    
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
        
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convertir formato de rostros
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x+w, y+h, x))  # (top, right, bottom, left)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detectar objetos
        frame, detections = detect_objects(frame, net, classes, face_locations)
        
        # Mostrar FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Verificar comportamientos peligrosos
        behaviors_detected = []
        
        for label, confidence in detections:
            # Verificar si pasó suficiente tiempo desde la última alerta
            if current_time - last_detection_time.get(label, 0) > cooldown_period:
                behaviors_detected.append(label)
                last_detection_time[label] = current_time
        
        # Mostrar alertas
        if behaviors_detected:
            # Overlay rojo
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Mostrar alertas en pantalla
            y_pos = 70
            for behavior in behaviors_detected:
                alert_text = f"¡ALERTA! Conducta peligrosa: {behavior}"
                cv2.putText(frame, alert_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
                
                # Reproducir alarma
                import threading
                threading.Thread(target=play_alarm).start()
                
                print(f"¡ALERTA! Se detectó: {behavior}")

        
       
        
        # Mostrar frame
        cv2.imshow("Detección de Comportamientos Peligrosos", frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Sistema de detección finalizado")

if __name__ == "__main__":
    main()