import cv2
import numpy as np
import time
from expresiones_faciales.config import ESTRES_CONFIG

class EstresAnalyzer:
    """Analizador de nivel de estrés basado en expresiones faciales"""
    
    def __init__(self):
        # Cargar configuración
        self.enabled = ESTRES_CONFIG["enabled"]
        self.sensitivity = ESTRES_CONFIG["sensitivity"]
        self.facial_tension_threshold = ESTRES_CONFIG["facial_tension_threshold"]
        self.micro_movement_threshold = ESTRES_CONFIG["micro_movement_threshold"]
        self.time_window_seconds = ESTRES_CONFIG["time_window_seconds"]
        
        # Estado interno
        self.landmark_history = []
        self.last_timestamps = []
        
    def update_config(self, new_config):
        """Actualiza la configuración en tiempo real"""
        for key, value in new_config.items():
            if key in ESTRES_CONFIG:
                ESTRES_CONFIG[key] = value
                
        # Actualizar variables locales
        self.enabled = ESTRES_CONFIG["enabled"]
        self.sensitivity = ESTRES_CONFIG["sensitivity"]
        self.facial_tension_threshold = ESTRES_CONFIG["facial_tension_threshold"]
        self.micro_movement_threshold = ESTRES_CONFIG["micro_movement_threshold"]
        self.time_window_seconds = ESTRES_CONFIG["time_window_seconds"]
        
    def analyze(self, frame, face_location):
        """Analiza nivel de estrés basado en expresiones faciales"""
        if not self.enabled:
            return 0.0
            
        # Extraer región facial
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        # Obtener landmarks faciales
        face_landmarks = self._get_landmarks(frame, face_location)
        if not face_landmarks:
            return 0.0
            
        # Medir tensión facial
        facial_tension = self._measure_facial_tension(face_landmarks)
        
        # Actualizar historial de landmarks
        current_time = time.time()
        if self.landmark_history:
            # Medir micro-movimientos
            micro_movements = self._measure_micro_movements(face_landmarks, self.landmark_history[-1])
        else:
            micro_movements = 0.0
            
        # Guardar en historial
        self.landmark_history.append(face_landmarks)
        self.last_timestamps.append(current_time)
        
        # Limpiar historial antiguo
        self._clean_old_history(current_time - self.time_window_seconds)
        
        # Calcular nivel de estrés (0.0-1.0)
        stress_level = self._calculate_stress_level(facial_tension, micro_movements)
        
        # Convertir a porcentaje (0-100)
        stress_percentage = int(stress_level * 100)
        
        return stress_percentage
    
    def _get_landmarks(self, frame, face_location):
        """Obtiene landmarks faciales"""
        # Implementación de extracción de landmarks
        # ...
        
    def _measure_facial_tension(self, landmarks):
        """Mide tensión facial basada en landmarks"""
        # Implementación de medición
        # ...
        
    def _measure_micro_movements(self, current_landmarks, previous_landmarks):
        """Mide micro-movimientos entre frames"""
        # Implementación de medición
        # ...
        
    def _clean_old_history(self, oldest_time):
        """Limpia registros más antiguos que oldest_time"""
        # Implementación de limpieza
        # ...
        
    def _calculate_stress_level(self, facial_tension, micro_movements):
        """Calcula nivel de estrés basado en métricas"""
        # Implementación de cálculo
        # ...