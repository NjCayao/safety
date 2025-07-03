import cv2
import numpy as np
import time
from expresiones_faciales.config import FATIGA_CONFIG

class FatigaDetector:
    """Detector de fatiga basado en expresiones faciales"""
    
    def __init__(self):
        # Cargar configuración
        self.enabled = FATIGA_CONFIG["enabled"]
        self.sensitivity = FATIGA_CONFIG["sensitivity"]
        self.blink_threshold = FATIGA_CONFIG["blink_threshold"]
        self.eye_aspect_ratio_threshold = FATIGA_CONFIG["eye_aspect_ratio_threshold"]
        self.time_window_seconds = FATIGA_CONFIG["time_window_seconds"]
        
        # Estado interno
        self.blink_history = []
        self.ear_history = []  # Eye Aspect Ratio history
        self.last_timestamps = []
        
    def update_config(self, new_config):
        """Actualiza la configuración en tiempo real"""
        for key, value in new_config.items():
            if key in FATIGA_CONFIG:
                FATIGA_CONFIG[key] = value
                
        # Actualizar variables locales
        self.enabled = FATIGA_CONFIG["enabled"]
        self.sensitivity = FATIGA_CONFIG["sensitivity"] 
        self.blink_threshold = FATIGA_CONFIG["blink_threshold"]
        self.eye_aspect_ratio_threshold = FATIGA_CONFIG["eye_aspect_ratio_threshold"]
        self.time_window_seconds = FATIGA_CONFIG["time_window_seconds"]
        
    def detect(self, frame, face_location):
        """Detecta nivel de fatiga basado en expresiones faciales"""
        if not self.enabled:
            return 0.0
            
        # Extraer región facial
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        # Obtener landmarks para ojos
        eye_landmarks = self._get_eye_landmarks(frame, face_location)
        if not eye_landmarks:
            return 0.0
            
        # Calcular EAR (Eye Aspect Ratio)
        left_ear = self._calculate_eye_aspect_ratio(eye_landmarks["left_eye"])
        right_ear = self._calculate_eye_aspect_ratio(eye_landmarks["right_eye"])
        ear = (left_ear + right_ear) / 2.0
        
        # Detectar parpadeo
        is_blink = ear < self.blink_threshold
        
        # Actualizar historiales
        current_time = time.time()
        self.ear_history.append(ear)
        self.last_timestamps.append(current_time)
        if is_blink:
            self.blink_history.append(current_time)
            
        # Limpiar historiales antiguos
        self._clean_old_history(current_time - self.time_window_seconds)
        
        # Calcular métricas de fatiga
        blink_frequency = self._calculate_blink_frequency()
        avg_ear = self._calculate_average_ear()
        
        # Calcular nivel de fatiga (0.0-1.0)
        fatigue_level = self._calculate_fatigue_level(blink_frequency, avg_ear)
        
        # Convertir a porcentaje (0-100)
        fatigue_percentage = int(fatigue_level * 100)
        
        return fatigue_percentage
    
    def _get_eye_landmarks(self, frame, face_location):
        """Obtiene landmarks de ojos"""
        # Implementación de extracción de landmarks de ojos
        # ...
        
    def _calculate_eye_aspect_ratio(self, eye_points):
        """Calcula la relación de aspecto del ojo (EAR)"""
        # Implementación de cálculo de EAR
        # ...
        
    def _clean_old_history(self, oldest_time):
        """Limpia registros más antiguos que oldest_time"""
        # Implementación de limpieza
        # ...
        
    def _calculate_blink_frequency(self):
        """Calcula frecuencia de parpadeo"""
        # Implementación de cálculo
        # ...
        
    def _calculate_average_ear(self):
        """Calcula EAR promedio"""
        # Implementación de cálculo
        # ...
        
    def _calculate_fatigue_level(self, blink_frequency, avg_ear):
        """Calcula nivel de fatiga basado en métricas"""
        # Implementación de cálculo
        # ...