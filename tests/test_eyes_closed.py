import cv2
import dlib
import numpy as np
import time
import traceback
from scipy.spatial import distance
import logging
import os
from datetime import datetime

class FatigueDetectionModule:
    def __init__(self, model_path, reports_dir="reports", eye_ar_thresh=0.17, eye_ar_consec_frames=10):
         
        # Para medición de tiempo más precisa
        self.frame_interval = 1/10  # Estimación de tiempo entre frames (asumiendo 30 FPS)
        self.time_with_eyes_closed = 0  # Tiempo acumulado con ojos cerrados
        self.target_closed_time = 1.5  # Tiempo objetivo (1.5 segundos)

        self.model_path = model_path
        self.reports_dir = reports_dir
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.eye_closed_counter = 0
        self.face_detector = None
        self.landmark_predictor = None
        self.logger = logging.getLogger('FatigueDetectionModule')
        
        # Para detección de bostezos
        self.yawn_threshold = 0.20  # Umbral reducido
        self.yawn_times = []  # Lista para registrar tiempos de bostezos
        self.last_yawn_time = 0  # Tiempo del último bostezo registrado
        self.yawn_min_count = 3  # Número mínimo de bostezos para alerta
        self.yawn_window = 600  # Ventana de 10 minutos en segundos
        
        # Para conteo de episodios de fatiga
        self.fatigue_episodes = []  # Lista para registrar tiempos de episodios de fatiga
        self.fatigue_threshold = 3  # Número de episodios para alerta grave
        self.fatigue_window = 600  # Ventana de 10 minutos en segundos
        
        # Para detección de distracciones
        self.distraction_counter = 0
        self.distraction_threshold = 45  # Frames consecutivos (aprox. 1.5 segundos)
        self.distraction_times = []
        self.distraction_min_count = 3  # Mínimo de distracciones para alerta
        
        # Asegurar que existe el directorio de reportes
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def initialize(self):
        """Inicializa el detector de fatiga"""
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(self.model_path)
            self.logger.info("Detector de landmarks faciales cargado correctamente")
            return True
        except Exception as e:
            self.logger.error(f"Error al cargar detector de landmarks: {str(e)}")
            return False
    
    def eye_aspect_ratio(self, eye):
        """Calcula el EAR (Eye Aspect Ratio)"""
        # Calcular distancias verticales
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Calcular distancia horizontal
        C = distance.euclidean(eye[0], eye[3])
        
        # Calcular EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_yawn(self, landmarks, frame):
        """Detecta bostezos basándose en la apertura de la boca"""
        try:
            # Obtener puntos de la boca
            mouth_points = []
            for i in range(48, 68):  # Landmarks de la boca
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                mouth_points.append((x, y))
                # Visualizar cada punto para depuración
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            # Dibujar contorno de la boca
            mouth_hull = cv2.convexHull(np.array(mouth_points))
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)
            
            # Calcular apertura vertical de la boca
            top_lip = mouth_points[13]  # Punto central del labio superior
            bottom_lip = mouth_points[19]  # Punto central del labio inferior

            # Visualizar los puntos clave
            cv2.circle(frame, top_lip, 3, (0, 0, 255), -1)  # Rojo para labio superior
            cv2.circle(frame, bottom_lip, 3, (255, 0, 0), -1)  # Azul para labio inferior
            
            # Calcular distancia vertical
            mouth_height = distance.euclidean(top_lip, bottom_lip)
            
            # Calcular ancho de la boca como referencia para normalizar
            left_mouth = mouth_points[0]
            right_mouth = mouth_points[6]

            # Visualizar puntos de anchura
            cv2.circle(frame, left_mouth, 3, (0, 255, 0), -1)  # Verde para esquina izquierda
            cv2.circle(frame, right_mouth, 3, (0, 255, 0), -1)  # Verde para esquina derecha    
            
            mouth_width = distance.euclidean(left_mouth, right_mouth)

            # Calcular ratio
            if mouth_width > 0:
                mouth_ratio = mouth_height / mouth_width
            else:
                mouth_ratio = 0

            # Mostrar medidas en pantalla
            cv2.putText(frame, f"Altura boca: {mouth_height:.1f}", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ancho boca: {mouth_width:.1f}", (10, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ratio: {mouth_ratio:.2f} (Umbral: {self.yawn_threshold:.2f})", (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Mostrar una barra de progreso visual para el ratio
            bar_length = 200
            bar_height = 20
            bar_x = 10
            bar_y = 340
            max_ratio = 0.5  # Reducido para que la barra sea más sensible 
            fill_width = int(bar_length * min(mouth_ratio/max_ratio, 1.0))
            
            # Barra base
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (200, 200, 200), -1)
            # Barra de progreso
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 255), -1)
            # Línea de umbral
            threshold_x = bar_x + int(bar_length * self.yawn_threshold/max_ratio)
            cv2.line(frame, (threshold_x, bar_y-5), (threshold_x, bar_y+bar_height+5), (0, 0, 255), 2)

            # Tiempo actual
            current_time = time.time()
            
            # Verificar si es un bostezo
            is_yawning = mouth_ratio > self.yawn_threshold
            if is_yawning:
                cv2.putText(frame, "BOSTEZO DETECTADO", (frame.shape[1]//2 - 150, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Solo considerar como un nuevo bostezo si ha pasado cierto tiempo desde el último
                if not hasattr(self, 'last_yawn_time') or (current_time - self.last_yawn_time > 3.0):
                    # Registrar tiempo del bostezo
                    self.yawn_times.append(current_time)
                    self.last_yawn_time = current_time
                    print(f"¡Nuevo bostezo registrado! Total: {len(self.yawn_times)}")
            
            # Eliminar bostezos antiguos (más de 10 minutos)
            ten_minutes_ago = current_time - self.yawn_window
            self.yawn_times = [t for t in self.yawn_times if t > ten_minutes_ago]
            
            # Mostrar contador de bostezos en 10 minutos
            cv2.putText(frame, f"Bostezos en 10 min: {len(self.yawn_times)}/{self.yawn_min_count}", 
                      (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Verificar si hay suficientes bostezos para una alerta
            multiple_yawns = len(self.yawn_times) >= self.yawn_min_count
            
            # Si hay múltiples bostezos, destacar visualmente
            if multiple_yawns:
                cv2.putText(frame, "¡ALERTA! MÚLTIPLES BOSTEZOS", (10, 420), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Resetear el contador después de la alerta
                self.yawn_times = []
            
            return is_yawning, multiple_yawns
            
        except Exception as e:
            self.logger.error(f"Error en detección de bostezos: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, False
    
    def detect_distraction(self, landmarks, frame):
        """Detecta si el operador está distraído (no mira al frente)"""
        try:
            # Puntos para los ojos y la nariz
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            
            # Puntos de la cabeza para referencia
            left_face = (landmarks.part(0).x, landmarks.part(0).y)
            right_face = (landmarks.part(16).x, landmarks.part(16).y)
            
            # Vector desde el centro de los ojos hacia la nariz
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            
            # Calcular ancho facial
            face_width = distance.euclidean(left_face, right_face)
            
            # Vector de dirección
            direction_vector = (nose[0] - eye_center[0], nose[1] - eye_center[1])
            
            # Medir la desviación horizontal (indica giro de cabeza)
            horizontal_deviation = abs(direction_vector[0])
            
            # Normalizar por ancho facial
            deviation_ratio = horizontal_deviation / face_width if face_width > 0 else 0
            
            # Mostrar indicador de desviación
            cv2.putText(frame, f"Desviación: {deviation_ratio:.2f}", (10, 480), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Dibujar línea de dirección de mirada
            cv2.line(frame, eye_center, nose, (0, 255, 0), 2)
            
            # Si la desviación es grande, el operador podría estar mirando a un lado
            # Umbral: aproximadamente el 15% del ancho facial
            is_distracted = deviation_ratio > 0.15
            
            # Contar frames consecutivos de distracción
            if is_distracted:
                self.distraction_counter += 1
            else:
                self.distraction_counter = 0
            
            # Verificar si la distracción persiste
            distraction_detected = self.distraction_counter >= self.distraction_threshold
            
            # Tiempo actual
            current_time = time.time()
            
            # Si se detecta distracción prolongada
            if distraction_detected:
                cv2.putText(frame, "¡DISTRACCIÓN DETECTADA!", (frame.shape[1]//2 - 200, frame.shape[0]//2 + 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # Registrar solo si han pasado al menos 5 segundos desde la última
                if not self.distraction_times or (current_time - self.distraction_times[-1] > 5.0):
                    self.distraction_times.append(current_time)
                    print(f"¡Nueva distracción detectada! Total: {len(self.distraction_times)}")
                
                # Reiniciar contador para no acumular
                self.distraction_counter = 0
            
            # Eliminar distracciones antiguas (más de 10 minutos)
            self.distraction_times = [t for t in self.distraction_times if t > current_time - self.fatigue_window]
            
            # Verificar si hay múltiples distracciones
            multiple_distractions = len(self.distraction_times) >= self.distraction_min_count
            
            # Mostrar contador de distracciones
            cv2.putText(frame, f"Distracciones: {len(self.distraction_times)}/{self.distraction_min_count}", 
                      (frame.shape[1] - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            if multiple_distractions:
                cv2.putText(frame, "¡ALERTA! MÚLTIPLES DISTRACCIONES", (10, 450), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # Resetear contador después de alerta
                self.distraction_times = []
            
            return distraction_detected, multiple_distractions
            
        except Exception as e:
            self.logger.error(f"Error en detección de distracción: {str(e)}")
            traceback.print_exc()
            return False, False
    
    def detect_fatigue(self, frame):
        """Detecta fatiga (ojos cerrados, bostezos) en el frame actual"""
        if self.face_detector is None or self.landmark_predictor is None:
            if not self.initialize():
                return False, False, False, False, False, False, frame                
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_detector(gray, 0)
        
        if len(faces) == 0:
            # Reiniciar contador si no se detecta rostro
            self.eye_closed_counter = 0
            return False, False, False, False, False, False, frame
        
        fatigue_detected = False
        multiple_fatigue_episodes = False
        is_yawning = False
        multiple_yawns = False
        distraction_detected = False
        multiple_distractions = False
        
        # Para cada rostro detectado (normalmente solo uno)
        # En el bucle de procesamiento de rostros:
        for face in faces:
            # Predecir landmarks faciales
            landmarks = self.landmark_predictor(gray, face)
            
            # Detección de distracciones (mirar a los lados)
            distraction_detected, multiple_distractions = self.detect_distraction(landmarks, frame)
            
            # Detección de bostezos
            is_yawning, multiple_yawns = self.detect_yawn(landmarks, frame)
            
            # Extraer coordenadas de los ojos
            left_eye = []
            right_eye = []
            
            # Ojo izquierdo (puntos 36-41)
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))
            
            # Ojo derecho (puntos 42-47)
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))
            
            # Calcular EAR para ambos ojos
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Promedio de EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Dibujar contornos de los ojos
            left_eye_hull = cv2.convexHull(np.array(left_eye))
            right_eye_hull = cv2.convexHull(np.array(right_eye))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Mostrar valor de EAR y umbral
            cv2.putText(frame, f"EAR: {ear:.2f} (Umbral: {self.eye_ar_thresh})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Verificar si los ojos están cerrados
            if ear < self.eye_ar_thresh:
                # Incrementar tiempo con ojos cerrados
                self.time_with_eyes_closed += self.frame_interval
                
                 # Mostrar tiempo con ojos cerrados
                cv2.putText(frame, f"Ojos cerrados: {self.time_with_eyes_closed:.1f}/{self.target_closed_time} seg", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                 # Reiniciar tiempo
                self.time_with_eyes_closed = 0
                
                # Mostrar contador normal
                cv2.putText(frame, f"Ojos cerrados: 0.0/{self.target_closed_time} seg", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Verificar si se detecta fatiga basado en tiempo
            if self.time_with_eyes_closed >= self.target_closed_time:
                # Tiempo actual
                current_time = time.time()
                
                # Registrar episodio de fatiga
                self.fatigue_episodes.append(current_time)
                
                # Eliminar episodios antiguos (más de 10 minutos)
                self.fatigue_episodes = [t for t in self.fatigue_episodes 
                                      if t > current_time - self.fatigue_window]
                
                # Mostrar contador de episodios de fatiga
                cv2.putText(frame, f"Episodios de fatiga: {len(self.fatigue_episodes)}/{self.fatigue_threshold}", 
                          (frame.shape[1] - 300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Verificar si hay múltiples episodios
                multiple_fatigue_episodes = len(self.fatigue_episodes) >= self.fatigue_threshold
                
                # Mostrar alerta visual
                cv2.putText(frame, "¡FATIGA DETECTADA!", (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Reiniciar contador
                self.time_with_eyes_closed = 0
                
                # Overlay rojo
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                fatigue_detected = True
                
                if multiple_fatigue_episodes:
                    cv2.putText(frame, "¡ALERTA! MÚLTIPLES EPISODIOS DE FATIGA", (10, 520), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Resetear después de alerta
                    self.fatigue_episodes = []
        
        return fatigue_detected, multiple_fatigue_episodes, is_yawning, multiple_yawns, distraction_detected, multiple_distractions, frame
    
    def generate_report(self, frame, alert_type, operator_info=None):
        """
        Genera un reporte para una alerta detectada
        
        Args:
            frame: Imagen del evento
            alert_type: Tipo de alerta (fatigue, multiple_fatigue, yawn, multiple_yawns, etc.)
            operator_info: Información del operador (opcional)
        """
        try:
            # Crear nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            operator_id = operator_info["id"] if operator_info else "unknown"
            filename = f"{operator_id}_{alert_type}_{timestamp}.jpg"
            
            # Guardar imagen
            image_path = os.path.join(self.reports_dir, filename)
            cv2.imwrite(image_path, frame)
            
            # Crear un archivo de texto con detalles adicionales
            details_file = os.path.join(self.reports_dir, f"{operator_id}_{alert_type}_{timestamp}.txt")
            with open(details_file, 'w') as f:
                f.write(f"Reporte de Seguridad - {alert_type.upper()}\n")
                f.write(f"Fecha y hora: {timestamp}\n")
                f.write(f"Operador: {operator_info['name'] if operator_info else 'Desconocido'}\n")
                
                if 'fatigue' in alert_type:
                    f.write(f"Episodios de fatiga en últimos 10 minutos: {len(self.fatigue_episodes)}\n")
                
                if 'yawn' in alert_type:
                    f.write(f"Bostezos en últimos 10 minutos: {len(self.yawn_times)}\n")
                
                if 'distraction' in alert_type:
                    f.write(f"Distracciones en últimos 10 minutos: {len(self.distraction_times)}\n")
                
                f.write("\nAcción recomendada: El operador debe tomar un descanso inmediato de al menos 15 minutos.\n")
            
            self.logger.info(f"Reporte generado: {image_path} y {details_file}")
            
            # Aquí podrías implementar el envío del reporte al servidor
            # self.send_report_to_server(image_path, details_file)
            
            return True
        except Exception as e:
            self.logger.error(f"Error al generar reporte: {str(e)}")
            traceback.print_exc()
            return False