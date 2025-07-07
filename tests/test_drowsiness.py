import cv2
import dlib
import numpy as np
import time
import os
from scipy.spatial import distance

# Constantes
EYE_AR_THRESH = 0.25  # Umbral para determinar si los ojos están cerrados
EYE_AR_CONSEC_FRAMES = 30  # Número de frames consecutivos para activar alarma

# Directorios
AUDIO_DIR = "audio"
MODELS_DIR = "models"

# Función para reproducir alarma usando playsound
def play_alarm(alarm_file="alarma.mp3"):
    """Intenta reproducir la alarma usando múltiples métodos"""
    methods_tried = 0
    alarm_played = False
    
    # Método 1: Pygame
    try:
        import pygame
        
        # Inicializar pygame mixer
        pygame.mixer.init()
        
        # Ruta al archivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        alarm_path = os.path.join(current_dir, AUDIO_DIR, alarm_file)
        alarm_path = os.path.normpath(alarm_path)
        
        print(f"Intentando reproducir con pygame: {alarm_path}")
        
        if os.path.exists(alarm_path):
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()
            print("Alarma reproducida con pygame")
            alarm_played = True
        
        methods_tried += 1
    except Exception as e:
        print(f"Método 1 (pygame) falló: {str(e)}")
    
    # Método 2: winsound.Beep
    if not alarm_played:
        try:
            import winsound
            print("Intentando reproducir con winsound.Beep")
            winsound.Beep(1000, 1000)  # 1000Hz durante 1 segundo
            print("Alarma reproducida con winsound.Beep")
            alarm_played = True
            
            methods_tried += 1
        except Exception as e:
            print(f"Método 2 (winsound.Beep) falló: {str(e)}")
    
    # Método 3: winsound.PlaySound con sonido del sistema
    if not alarm_played:
        try:
            import winsound
            print("Intentando reproducir sonido del sistema")
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            print("Alarma del sistema reproducida")
            alarm_played = True
            
            methods_tried += 1
        except Exception as e:
            print(f"Método 3 (sonido del sistema) falló: {str(e)}")
    
    # Método 4: playsound como último recurso
    if not alarm_played:
        try:
            from playsound import playsound
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alarm_path = os.path.join(current_dir, AUDIO_DIR, "alarma.mp3")
            
            print(f"Intentando reproducir con playsound: {alarm_path}")
            
            if os.path.exists(alarm_path):
                playsound(alarm_path, block=False)
                print("Alarma reproducida con playsound")
                alarm_played = True
            
            methods_tried += 1
        except Exception as e:
            print(f"Método 4 (playsound) falló: {str(e)}")
    
    # Resumen
    if alarm_played:
        print(f"Alarma reproducida exitosamente después de probar {methods_tried} métodos")
    else:
        print(f"ADVERTENCIA: No se pudo reproducir la alarma después de probar {methods_tried} métodos")

# Función para calcular el ratio de aspecto del ojo (EAR)
def eye_aspect_ratio(eye):
    # Calcular distancias verticales
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Calcular distancia horizontal
    C = distance.euclidean(eye[0], eye[3])
    
    # Calcular EAR
    ear = (A + B) / (2.0 * C)
    return ear

def main():
    print("Iniciando detector de somnolencia...")
    
    # Verificar directorios
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
        print(f"Carpeta de audio creada: {AUDIO_DIR}")
        print("Coloca tus archivos de audio de alarma en esta carpeta")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Carpeta de modelos creada: {MODELS_DIR}")
    
    # Cargar detector facial y predictor de landmarks
    detector = dlib.get_frontal_face_detector()
    
    # Ruta al modelo de landmarks
    model_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
    
    try:
        predictor = dlib.shape_predictor(model_path)
        print("Modelo de landmarks faciales cargado correctamente")
    except RuntimeError as e:
        print(f"Error al cargar el modelo: {e}")
        print(f"Asegúrate de tener el archivo en: {model_path}")
        print("Descárgalo desde: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    # Inicializar contador de frames
    counter = 0
    alarm_active = False
    last_alarm_time = 0
    
    print("Detector de somnolencia activo. Presiona 'q' para salir.")
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        # Redimensionar frame para mejor rendimiento
        frame = cv2.resize(frame, (640, 480))
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = detector(gray, 0)
        
        if len(faces) == 0:
            cv2.putText(frame, "No se detecta rostro", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for face in faces:
            # Detectar landmarks faciales
            landmarks = predictor(gray, face)
            
            # Extraer coordenadas de los ojos
            left_eye = []
            right_eye = []
            
            # Ojo izquierdo (puntos 36-41)
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Ojo derecho (puntos 42-47)
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Calcular EAR para ambos ojos
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Promedio de EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Dibujar contorno de los ojos
            left_hull = cv2.convexHull(np.array(left_eye))
            right_hull = cv2.convexHull(np.array(right_eye))
            cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)
            
            # Mostrar valor de EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Verificar si los ojos están cerrados
            if ear < EYE_AR_THRESH:
                counter += 1
                
                # Mostrar contador
                cv2.putText(frame, f"Ojos cerrados: {counter}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Verificar si se detecta somnolencia
                if counter >= EYE_AR_CONSEC_FRAMES:
                    # Activar alarma con un intervalo mínimo de 3 segundos
                    current_time = time.time()
                    if current_time - last_alarm_time > 3:
                        alarm_active = True
                        last_alarm_time = current_time
                        # Reproducir alarma en un hilo separado
                        import threading
                        threading.Thread(target=play_alarm, args=("alarma.mp3",)).start()
                    
                    # Mostrar alerta en pantalla
                    cv2.putText(frame, "¡ALERTA! SOMNOLENCIA", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Hacer que el frame parpadee en rojo
                    red_overlay = np.zeros_like(frame)
                    red_overlay[:] = (0, 0, 200)  # Color rojo
                    cv2.addWeighted(red_overlay, 0.3, frame, 0.7, 0, frame)
            else:
                # Reiniciar contador si los ojos están abiertos
                counter = 0
                alarm_active = False
        
        # Mostrar frame
        cv2.imshow("Detector de Somnolencia", frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Detector de somnolencia finalizado")

if __name__ == "__main__":
    main()