import cv2
import numpy as np
import os
import time

# Configuración
MODEL_DIR = "models"
CONFIG_FILE = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3.weights")
CLASSES_FILE = os.path.join(MODEL_DIR, "coco.names")

# Clases de interés (COCO dataset)
TARGET_CLASSES = ["cell phone", "cigarette"]

def load_model():
    """Carga el modelo YOLO para detección de objetos"""
    print("Cargando modelo de detección de objetos...")
    
    # Verificar si existen los archivos necesarios
    if not os.path.exists(CONFIG_FILE) or not os.path.exists(WEIGHTS_FILE):
        print("Archivos del modelo no encontrados. Debes descargar:")
        print(f"1. {CONFIG_FILE}")
        print(f"2. {WEIGHTS_FILE}")
        print("Puedes descargarlos desde: https://pjreddie.com/darknet/yolo/")
        return None
    
    # Cargar clases
    if not os.path.exists(CLASSES_FILE):
        print(f"Archivo de clases no encontrado: {CLASSES_FILE}")
        return None
    
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Cargar modelo
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    
    # Configurar backend (CPU/GPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    print("Modelo cargado correctamente")
    return net, classes

def detect_objects(frame, net, classes):
    """Detecta objetos en el frame usando YOLO"""
    height, width = frame.shape[:2]
    
    # Crear blob desde la imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Pasar el blob por la red
    net.setInput(blob)
    
    # Obtener las capas de salida
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    
    # Inicializar listas para las detecciones
    boxes = []
    confidences = []
    class_ids = []
    
    # Para cada detección
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filtrar por confianza y clases de interés
            if confidence > 0.5 and classes[class_id] in TARGET_CLASSES:
                # Coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas del rectángulo
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Aplicar non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detections = []
    colors = [(0, 0, 255), (0, 255, 255)]  # Rojo para celular, amarillo para cigarrillo
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[0] if label == "cell phone" else colors[1]
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Añadir etiqueta
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Añadir a la lista de detecciones
            detections.append((label, confidence))
    
    return frame, detections

def main():
    """Función principal para probar la detección de comportamientos peligrosos"""
    # Crear directorio de modelos si no existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Cargar modelo
    model_data = load_model()
    if model_data is None:
        return
    
    net, classes = model_data
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    print("Prueba de detección de comportamientos iniciada. Presione 'q' para salir.")
    
    # Para medir FPS
    prev_time = 0
    
    # Para evitar alertas repetitivas
    last_detection_time = {"cell phone": 0, "cigarette": 0}
    cooldown_period = 3  # Segundos entre alertas del mismo tipo
    
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
        
        # Mostrar FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detectar objetos
        frame, detections = detect_objects(frame, net, classes)
        
        # Verificar comportamientos peligrosos y mostrar alertas
        behaviors_detected = []
        
        for label, confidence in detections:
            # Verificar cooldown para evitar alertas repetitivas
            if current_time - last_detection_time.get(label, 0) > cooldown_period:
                behaviors_detected.append(label)
                last_detection_time[label] = current_time
        
        # Mostrar alertas
        if behaviors_detected:
            # Overlay rojo en caso de comportamiento peligroso
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Mostrar alertas
            y_pos = 70
            for behavior in behaviors_detected:
                alert_text = f"¡ALERTA! Conducta peligrosa: {behavior}"
                cv2.putText(frame, alert_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
                
                # Aquí se reproduciría la alarma sonora
                print(f"¡ALERTA! Se detectó: {behavior}")
        
        # Mostrar frame
        cv2.imshow("Detección de Comportamientos", frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Prueba de detección de comportamientos finalizada")

if __name__ == "__main__":
    main()