"""
Módulo de Análisis de Emociones Faciales
========================================
Detecta y analiza expresiones emocionales en tiempo real.
"""

import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import logging

class EmotionAnalyzer:
    def __init__(self):
        """Inicializa el analizador de emociones"""
        self.logger = logging.getLogger('EmotionAnalyzer')
        
        # Historial para estabilizar detecciones
        self.emotion_history = deque(maxlen=10)
        self.current_emotion = "neutral"
        
        # Configuración de sensibilidad
        self.sensitivity = 0.7
        self.confidence_threshold = 0.65
        
        # Colores para cada emoción (BGR)
        self.emotion_colors = {
            'neutral': (200, 200, 200),   # Gris
            'sonrisa': (0, 255, 0),       # Verde
            'tristeza': (255, 0, 0),      # Azul
            'sorpresa': (0, 255, 255),    # Amarillo
            'enojo': (0, 0, 255),         # Rojo
            'concentrado': (255, 165, 0)  # Naranja
        }
        
        # Métricas de la última detección
        self.last_metrics = {}
        
    def analyze(self, frame, face_landmarks):
        """
        Analiza la expresión facial basada en landmarks.
        
        Args:
            frame: Imagen del rostro
            face_landmarks: Diccionario con puntos faciales
            
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'metrics': dict,
                'color': tuple
            }
        """
        if not face_landmarks:
            return self._default_result()
        
        try:
            # Calcular métricas faciales
            metrics = self._calculate_facial_metrics(face_landmarks)
            self.last_metrics = metrics
            
            # Clasificar emoción
            emotion, confidence = self._classify_emotion(metrics)
            
            # Agregar al historial
            self.emotion_history.append(emotion)
            
            # Obtener emoción estable
            stable_emotion = self._get_stable_emotion()
            
            return {
                'emotion': stable_emotion,
                'confidence': confidence,
                'metrics': metrics,
                'color': self.emotion_colors.get(stable_emotion, (200, 200, 200))
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando emoción: {str(e)}")
            return self._default_result()
    
    def _calculate_facial_metrics(self, landmarks):
        """Calcula métricas faciales para determinar expresiones"""
        metrics = {}
        
        # 1. Métrica de sonrisa (curvatura de la boca)
        mouth_left = landmarks['top_lip'][0]
        mouth_right = landmarks['top_lip'][6]
        mouth_center = landmarks['top_lip'][3]
        
        # Calcular curvatura
        mouth_width = self._distance(mouth_left, mouth_right)
        mouth_curve = self._calculate_curve(mouth_left, mouth_center, mouth_right)
        metrics['smile_ratio'] = mouth_curve / mouth_width if mouth_width > 0 else 0
        
        # 2. Apertura de boca (sorpresa)
        top_lip_center = self._midpoint(landmarks['top_lip'][3], landmarks['top_lip'][4])
        bottom_lip_center = self._midpoint(landmarks['bottom_lip'][3], landmarks['bottom_lip'][4])
        mouth_opening = self._distance(top_lip_center, bottom_lip_center)
        metrics['mouth_opening_ratio'] = mouth_opening / mouth_width if mouth_width > 0 else 0
        
        # 3. Elevación de cejas (sorpresa/alegría)
        left_eyebrow_height = self._average_height(landmarks['left_eyebrow'])
        right_eyebrow_height = self._average_height(landmarks['right_eyebrow'])
        left_eye_height = self._average_height(landmarks['left_eye'])
        right_eye_height = self._average_height(landmarks['right_eye'])
        
        eyebrow_eye_distance = ((left_eyebrow_height - left_eye_height) + 
                                (right_eyebrow_height - right_eye_height)) / 2
        
        face_height = self._distance(landmarks['chin'][8], landmarks['left_eyebrow'][2])
        metrics['eyebrow_raise_ratio'] = eyebrow_eye_distance / face_height if face_height > 0 else 0
        
        # 4. Fruncimiento de ceño (enojo/concentración)
        eyebrow_distance = self._distance(
            landmarks['left_eyebrow'][4],
            landmarks['right_eyebrow'][0]
        )
        metrics['eyebrow_distance_ratio'] = eyebrow_distance / mouth_width if mouth_width > 0 else 0
        
        # 5. Comisuras de la boca (tristeza)
        mouth_corner_left_height = mouth_left[1]
        mouth_corner_right_height = mouth_right[1]
        mouth_center_height = mouth_center[1]
        
        corner_drop = ((mouth_corner_left_height + mouth_corner_right_height) / 2) - mouth_center_height
        metrics['mouth_corner_drop'] = corner_drop / face_height if face_height > 0 else 0
        
        # 6. Apertura de ojos (cansancio/tristeza)
        left_eye_opening = self._calculate_eye_opening(landmarks['left_eye'])
        right_eye_opening = self._calculate_eye_opening(landmarks['right_eye'])
        metrics['eye_opening_ratio'] = (left_eye_opening + right_eye_opening) / 2
        
        return metrics
    
    def _classify_emotion(self, metrics):
        """Clasifica la emoción basada en las métricas"""
        emotion = "neutral"
        confidence = 0.6
        
        # Reglas de clasificación basadas en investigación
        
        # SONRISA/FELICIDAD
        if (metrics['smile_ratio'] > 0.03 and 
            metrics['mouth_corner_drop'] < -0.01):
            emotion = "sonrisa"
            confidence = min(0.9, 0.6 + metrics['smile_ratio'] * 10)
        
        # SORPRESA
        elif (metrics['mouth_opening_ratio'] > 0.15 and 
              metrics['eyebrow_raise_ratio'] > 0.12):
            emotion = "sorpresa"
            confidence = 0.8
        
        # TRISTEZA
        elif (metrics['mouth_corner_drop'] > 0.02 and 
              metrics['eye_opening_ratio'] < 0.22):
            emotion = "tristeza"
            confidence = 0.75
        
        # ENOJO
        elif (metrics['eyebrow_distance_ratio'] < 0.9 and 
              metrics['mouth_corner_drop'] > 0.01):
            emotion = "enojo"
            confidence = 0.7
        
        # CONCENTRADO
        elif (metrics['eyebrow_distance_ratio'] < 0.95 and 
              metrics['eye_opening_ratio'] < 0.25):
            emotion = "concentrado"
            confidence = 0.7
        
        return emotion, confidence
    
    def _get_stable_emotion(self):
        """Obtiene la emoción más estable del historial"""
        if not self.emotion_history:
            return "neutral"
        
        # Contar ocurrencias
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Obtener la más común
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        
        # Requerir al menos 40% de consenso
        if most_common[1] >= len(self.emotion_history) * 0.4:
            return most_common[0]
        
        return self.current_emotion
    
    def draw_emotion_info(self, frame, emotion_data, position=(10, 30)):
        """Dibuja información de emoción en el frame"""
        if not emotion_data:
            return frame
        
        emotion = emotion_data['emotion']
        confidence = emotion_data['confidence']
        color = emotion_data['color']
        
        # Texto principal
        text = f"Emocion: {emotion.upper()}"
        cv2.putText(frame, text, position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barra de confianza
        bar_width = 200
        bar_height = 10
        bar_x = position[0]
        bar_y = position[1] + 25
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Barra de confianza
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + confidence_width, bar_y + bar_height),
                     color, -1)
        
        # Texto de confianza
        conf_text = f"Confianza: {confidence:.0%}"
        cv2.putText(frame, conf_text, 
                   (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_emotion_summary(self, time_window=60):
        """Obtiene un resumen de emociones en la ventana de tiempo"""
        # Esta función podría expandirse para análisis temporal
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(self.emotion_history)
        if total == 0:
            return {}
        
        summary = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return summary
    
    # Métodos auxiliares
    def _distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _midpoint(self, p1, p2):
        """Calcula punto medio entre dos puntos"""
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    
    def _calculate_curve(self, p1, p2, p3):
        """Calcula la curvatura de tres puntos"""
        # Usa el área del triángulo como medida de curvatura
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                  (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2.0
        return area
    
    def _average_height(self, points):
        """Calcula altura promedio de un conjunto de puntos"""
        if not points:
            return 0
        return sum(p[1] for p in points) / len(points)
    
    def _calculate_eye_opening(self, eye_points):
        """Calcula apertura del ojo (EAR - Eye Aspect Ratio)"""
        if len(eye_points) != 6:
            return 0.3
        
        # Distancias verticales
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        
        # Distancia horizontal
        h = self._distance(eye_points[0], eye_points[3])
        
        # EAR
        if h == 0:
            return 0
        return (v1 + v2) / (2.0 * h)
    
    def _default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'metrics': {},
            'color': self.emotion_colors['neutral']
        }
    
    def reset(self):
        """Reinicia el analizador"""
        self.emotion_history.clear()
        self.current_emotion = "neutral"
        self.last_metrics = {}