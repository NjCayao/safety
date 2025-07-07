import cv2
import numpy as np
import os

def test_yolo():
    # Cargar YOLO
    net = cv2.dnn.readNetFromDarknet(
        "assets/models/yolov3-tiny.cfg",
        "assets/models/yolov3-tiny.weights"
    )
    
    # Cargar nombres de clases
    with open("assets/models/coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"Total clases: {len(classes)}")
    
    # Buscar índice de "cell phone"
    for i, name in enumerate(classes):
        if "phone" in name.lower() or "cell" in name.lower():
            print(f"Clase teléfono encontrada: '{name}' en índice {i}")
    
    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preparar imagen
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Obtener detecciones
        output_layers = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers)
        
        # Procesar detecciones
        height, width = frame.shape[:2]
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:
                    # Obtener coordenadas
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    # Dibujar detección
                    color = (0, 255, 0) if classes[class_id] == "cell phone" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    print(f"Detectado: {classes[class_id]} con {confidence:.2f}")
        
        cv2.imshow("Test YOLO", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolo()