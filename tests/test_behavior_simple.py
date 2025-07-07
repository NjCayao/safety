import cv2
import numpy as np
import os
import time

# Directorio para archivos de audio
AUDIO_DIR = "audio"

# Función para reproducir alarma
def play_alarm(alarm_file="alarma.mp3"):
    try:
        import pygame
        
        # Inicializar pygame
        pygame.mixer.init()
        
        # Construir ruta al archivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        alarm_path = os.path.join(current_dir, AUDIO_DIR, alarm_file)
        alarm_path = os.path.normpath(alarm_path)
        
        # Verificar si existe
        if os.path.exists(alarm_path):
            print(f"Reproduciendo alarma: {alarm_path}")
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()
        else:
            print(f"Archivo de audio no encontrado: {alarm_path}")
            
    except Exception as e:
        print(f"Error al reproducir alarma: {str(e)}")

def simulate_detection(frame):
    """
    Simula la detección de objetos (teléfono o cigarrillo)
    Esta función detecta basándose en colores para simular la detección
    """
    # Convertir a HSV para mejor detección de colores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir rango de colores para "teléfono" (azul)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Definir rango de colores para "cigarrillo" (rojo/naranja)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Crear máscaras
    phone_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    smoke_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Encontrar contornos
    phone_contours, _ = cv2.findContours(phone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    # Procesar contornos de "teléfono"
    for contour in phone_contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filtrar pequeños ruidos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Telefono", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            detections.append(("phone", 0.95))
            break  # Detectar solo uno
    
    # Procesar contornos de "cigarrillo"
    for contour in smoke_contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filtrar pequeños ruidos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Cigarrillo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detections.append(("smoking", 0.90))
            break  # Detectar solo uno
    
    return frame, detections

def main():
    """Función principal para probar la detección de comportamientos peligrosos"""
    # Crear directorios si no existen
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Mensaje de instrucciones
    print("=== Detector de Comportamientos Peligrosos (Simulado) ===")
    print("Instrucciones:")
    print("1. Muestre un objeto AZUL frente a la cámara para simular un teléfono")
    print("2. Muestre un objeto ROJO frente a la cámara para simular un cigarrillo")
    print("3. Presione 'q' para salir")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    # Para evitar alertas repetitivas
    last_detection_time = {"phone": 0, "smoking": 0}
    cooldown_period = 3  # Segundos entre alertas del mismo tipo
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        # Tiempo actual
        current_time = time.time()
        
        # Detectar objetos (simulado)
        frame, detections = simulate_detection(frame)
        
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
                
                # Reproducir alarma
                import threading
                threading.Thread(target=play_alarm).start()
                
                # Mostrar en consola
                print(f"¡ALERTA! Se detectó: {behavior}")
        
        # Mostrar instrucciones en pantalla
        cv2.putText(frame, "Azul = Telefono, Rojo = Cigarrillo", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Deteccion de Comportamientos", frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Prueba finalizada")

if __name__ == "__main__":
    main()