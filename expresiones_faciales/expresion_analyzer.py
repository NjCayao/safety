import cv2
import numpy as np
import face_recognition
from expresiones_faciales.config import EXPRESSION_CONFIG

class ExpresionAnalyzer:
    """Analizador de expresiones faciales"""
    
    def __init__(self):
        # Cargar configuración
        self.enabled = EXPRESSION_CONFIG["enabled"]
        self.sensitivity = EXPRESSION_CONFIG["sensitivity"]
        self.memory_frames = EXPRESSION_CONFIG["memory_frames"]
        self.expression_threshold = EXPRESSION_CONFIG["expression_threshold"]
        
        # Estado interno
        self.expression_history = []
        self.current_expression = "neutral"
        
    def update_config(self, new_config):
        """Actualiza la configuración en tiempo real"""
        for key, value in new_config.items():
            if key in EXPRESSION_CONFIG:
                EXPRESSION_CONFIG[key] = value
                
        # Actualizar variables locales
        self.enabled = EXPRESSION_CONFIG["enabled"]
        self.sensitivity = EXPRESSION_CONFIG["sensitivity"]
        self.memory_frames = EXPRESSION_CONFIG["memory_frames"]
        self.expression_threshold = EXPRESSION_CONFIG["expression_threshold"]
        
    def analyze(self, frame, face_location):
        """Analiza la expresión facial en un frame"""
        if not self.enabled:
            return {"expression": "disabled", "confidence": 0.0}
            
        # Extraer región facial
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        # Obtener landmarks faciales detallados
        face_landmarks = self._get_landmarks(frame, face_location)
        if not face_landmarks:
            return {"expression": "unknown", "confidence": 0.0}
        
        # Calcular métricas para diferentes expresiones
        metrics = self._calculate_expression_metrics(face_landmarks)
        
        # Clasificar expresión
        expression, confidence = self._classify_expression(metrics)
        
        # Actualizar historial
        self.expression_history.append(expression)
        if len(self.expression_history) > self.memory_frames:
            self.expression_history.pop(0)
        
        # Determinar expresión final (con filtrado)
        self.current_expression = self._get_stable_expression()
        
        return {
            "expression": self.current_expression,
            "confidence": confidence,
            "raw_metrics": metrics
        }
    
    def _get_landmarks(self, frame, face_location):
        """Obtiene landmarks faciales detallados"""
        # Implementación de extracción de landmarks
        # ...
        
    def _calculate_expression_metrics(self, landmarks):
        """Calcula métricas para diferentes expresiones"""
        # Implementación de cálculo de métricas
        # ...
        
    def _classify_expression(self, metrics):
        """Clasifica la expresión facial basada en métricas"""
        # Implementación de clasificación
        # ...
        
    def _get_stable_expression(self):
        """Obtiene expresión estable basada en historial"""
        # Implementación de estabilización
        # ...