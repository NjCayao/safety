"""
Módulo de Análisis de Emociones Faciales - Versión Optimizada
=============================================================
Detecta y analiza expresiones emocionales con soporte headless.
"""

import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import logging
import time

class EmotionAnalyzer:
    def __init__(self, headless=False):
        """
        Inicializa el analizador de emociones.
        
        Args:
            headless: Si True, desactiva visualizaciones (modo servidor)
        """
        self.logger = logging.getLogger('EmotionAnalyzer')
        self.headless = headless
        
        # Historial para estabilizar detecciones
        self.emotion_history = deque(maxlen=10)
        self.current_emotion = "neutral"
        
        # Configuración de sensibilidad
        self.sensitivity = 0.7
        self.confidence_threshold = 0.65
        
        # Baseline personalizado
        self.baseline = None
        self.is_calibrated = False
        
        # Colores para cada emoción (BGR) - solo si no es headless
        if not self.headless:
            self.emotion_colors = {
                'neutral': (200, 200, 200),   # Gris
                'happy': (0, 255, 0),         # Verde
                'sad': (255, 0, 0),           # Azul
                'angry': (0, 0, 255),         # Rojo
                'surprised': (255, 255, 0),   # Amarillo
                'fear': (255, 0, 255),        # Magenta
                'disgust': (0, 100, 255)      # Rojo oscuro
            }
        
        # Métricas de la última detección
        self.last_metrics = {}
        self.last_update = time.time()
        
    def set_baseline(self, baseline):
        """
        Configura el baseline del operador actual.
        
        Args:
            baseline: Diccionario con métricas calibradas del operador
        """
        if baseline and 'facial_metrics' in baseline:
            self.baseline = baseline
            self.is_calibrated = True
            self.logger.info("Baseline de emociones configurado")
    
    def analyze(self, frame, face_landmarks):
        """
        Analiza la expresión facial basada en landmarks.
        
        Args:
            frame: Imagen del rostro
            face_landmarks: Diccionario con puntos faciales
            
        Returns:
            dict: Resultados del análisis emocional
        """
        self.last_update = time.time()
        
        if not face_landmarks:
            return self._default_result()
        
        try:
            # Calcular métricas faciales
            metrics = self._calculate_facial_metrics(face_landmarks)
            self.last_metrics = metrics
            
            # Ajustar métricas según baseline si está calibrado
            if self.is_calibrated and self.baseline:
                metrics = self._adjust_metrics_to_baseline(metrics)
            
            # Clasificar emociones
            emotions = self._classify_emotions(metrics)
            
            # Calcular métricas derivadas
            wellbeing = self._calculate_wellbeing(emotions)
            valence = self._calculate_valence(emotions)  # Positivo/Negativo
            arousal = self._calculate_arousal(emotions)  # Activación/Calma
            
            # Determinar emoción dominante
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Agregar al historial
            self.emotion_history.append(dominant_emotion)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'wellbeing': wellbeing,
                'valence': valence,
                'arousal': arousal,
                'confidence': self._calculate_confidence(emotions),
                'is_calibrated': self.is_calibrated
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando emoción: {str(e)}")
            return self._default_result()
    
    def _adjust_metrics_to_baseline(self, metrics):
        """Ajusta métricas según el baseline calibrado del operador"""
        if not self.baseline:
            return metrics
        
        adjusted = metrics.copy()
        facial_metrics = self.baseline.get('facial_metrics', {})
        
        # Ajustar apertura de ojos
        if 'eye_measurements' in facial_metrics:
            baseline_eye = facial_metrics['eye_measurements'].get('eye_openness_avg', 0.25)
            if baseline_eye > 0:
                adjusted['eye_opening_ratio'] = metrics['eye_opening_ratio'] / baseline_eye
        
        # Ajustar distancia de cejas
        if 'eyebrow_measurements' in facial_metrics:
            baseline_brow = facial_metrics['eyebrow_measurements'].get('eyebrow_distance_avg', 1.0)
            if baseline_brow > 0:
                adjusted['eyebrow_distance_ratio'] = metrics['eyebrow_distance_ratio'] / baseline_brow
        
        # Ajustar ratio de boca
        if 'mouth_measurements' in facial_metrics:
            baseline_mouth = facial_metrics['mouth_measurements'].get('mouth_ratio_avg', 0.1)
            # Para sonrisa, comparar desviación del baseline
            adjusted['smile_deviation'] = abs(metrics['smile_ratio'] - baseline_mouth)
        
        return adjusted
    
    def _calculate_facial_metrics(self, landmarks):
        """Calcula métricas faciales optimizadas"""
        metrics = {}
        
        # 1. Sonrisa (curvatura de labios)
        mouth_left = landmarks['top_lip'][0]
        mouth_right = landmarks['top_lip'][6]
        mouth_center = landmarks['top_lip'][3]
        
        mouth_width = self._distance(mouth_left, mouth_right)
        corner_height_avg = (mouth_left[1] + mouth_right[1]) / 2
        smile_curve = (mouth_center[1] - corner_height_avg) / mouth_width if mouth_width > 0 else 0
        metrics['smile_ratio'] = smile_curve
        
        # 2. Apertura de boca
        top_lip_center = landmarks['top_lip'][3]
        bottom_lip_center = landmarks['bottom_lip'][3]
        mouth_opening = self._distance(top_lip_center, bottom_lip_center)
        metrics['mouth_opening_ratio'] = mouth_opening / mouth_width if mouth_width > 0 else 0
        
        # 3. Elevación de cejas
        left_eyebrow_avg = np.mean([p[1] for p in landmarks['left_eyebrow']])
        right_eyebrow_avg = np.mean([p[1] for p in landmarks['right_eyebrow']])
        left_eye_avg = np.mean([p[1] for p in landmarks['left_eye']])
        right_eye_avg = np.mean([p[1] for p in landmarks['right_eye']])
        
        eyebrow_raise_left = abs(left_eyebrow_avg - left_eye_avg)
        eyebrow_raise_right = abs(right_eyebrow_avg - right_eye_avg)
        metrics['eyebrow_raise_ratio'] = (eyebrow_raise_left + eyebrow_raise_right) / 2
        
        # 4. Fruncimiento de ceño
        left_inner = landmarks['left_eyebrow'][-1]
        right_inner = landmarks['right_eyebrow'][0]
        eyebrow_distance = self._distance(left_inner, right_inner)
        
        # Normalizar por distancia entre ojos
        left_eye_center = self._get_center(landmarks['left_eye'])
        right_eye_center = self._get_center(landmarks['right_eye'])
        eye_distance = self._distance(left_eye_center, right_eye_center)
        metrics['eyebrow_distance_ratio'] = eyebrow_distance / eye_distance if eye_distance > 0 else 1.0
        
        # 5. Apertura de ojos
        left_ear = self._calculate_ear(landmarks['left_eye'])
        right_ear = self._calculate_ear(landmarks['right_eye'])
        metrics['eye_opening_ratio'] = (left_ear + right_ear) / 2
        
        # 6. Asimetría facial
        metrics['facial_asymmetry'] = self._calculate_facial_asymmetry(landmarks)
        
        return metrics
    
    def _classify_emotions(self, metrics):
        """Clasifica emociones usando el modelo circumplex simplificado"""
        emotions = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprised': 0,
            'fear': 0,
            'disgust': 0,
            'neutral': 60  # Base
        }
        
        # FELICIDAD - Sonrisa + ojos ligeramente cerrados
        if metrics['smile_ratio'] > 0.03:
            intensity = min(1.0, metrics['smile_ratio'] / 0.08)
            emotions['happy'] = int(30 + intensity * 50)
            emotions['neutral'] -= 20
        
        # TRISTEZA - Comisuras caídas + cejas caídas
        if metrics['smile_ratio'] < -0.01:
            intensity = min(1.0, abs(metrics['smile_ratio']) / 0.05)
            emotions['sad'] = int(20 + intensity * 40)
            emotions['neutral'] -= 15
        
        # IRA - Cejas juntas + tensión
        if metrics['eyebrow_distance_ratio'] < 0.85:
            intensity = min(1.0, (0.9 - metrics['eyebrow_distance_ratio']) / 0.2)
            emotions['angry'] = int(20 + intensity * 50)
            emotions['neutral'] -= 15
        
        # SORPRESA - Ojos muy abiertos + boca abierta
        if metrics['eye_opening_ratio'] > 0.35 and metrics['mouth_opening_ratio'] > 0.15:
            emotions['surprised'] = 50
            emotions['neutral'] -= 20
        
        # MIEDO - Similar a sorpresa pero con cejas levantadas
        if metrics['eye_opening_ratio'] > 0.32 and metrics['eyebrow_raise_ratio'] > 30:
            emotions['fear'] = 40
            emotions['neutral'] -= 15
        
        # ASCO - Asimetría facial + labio superior levantado
        if metrics['facial_asymmetry'] > 0.3:
            emotions['disgust'] = 30
            emotions['neutral'] -= 10
        
        # Normalizar a 100%
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] = int((emotions[emotion] / total) * 100)
        
        return emotions
    
    def _calculate_wellbeing(self, emotions):
        """Calcula bienestar general (0-100)"""
        positive = emotions.get('happy', 0) + emotions.get('neutral', 0) * 0.5
        negative = (emotions.get('sad', 0) + emotions.get('angry', 0) + 
                   emotions.get('fear', 0) + emotions.get('disgust', 0))
        
        total = positive + negative
        if total > 0:
            wellbeing = (positive / total) * 100
        else:
            wellbeing = 50
        
        return int(wellbeing)
    
    def _calculate_valence(self, emotions):
        """Calcula valencia emocional (-100 a +100)"""
        positive = emotions.get('happy', 0)
        negative = emotions.get('sad', 0) + emotions.get('angry', 0) + emotions.get('disgust', 0)
        
        valence = positive - negative
        return max(-100, min(100, valence))
    
    def _calculate_arousal(self, emotions):
        """Calcula nivel de activación emocional (0-100)"""
        high_arousal = (emotions.get('angry', 0) + emotions.get('surprised', 0) + 
                       emotions.get('fear', 0) + emotions.get('happy', 0) * 0.7)
        low_arousal = emotions.get('sad', 0) * 0.3 + emotions.get('neutral', 0) * 0.1
        
        arousal = high_arousal - low_arousal
        return max(0, min(100, arousal))
    
    def _calculate_confidence(self, emotions):
        """Calcula confianza en la detección"""
        # Mayor diferencia entre primera y segunda emoción = mayor confianza
        sorted_emotions = sorted(emotions.values(), reverse=True)
        if len(sorted_emotions) >= 2 and sorted_emotions[1] > 0:
            confidence = (sorted_emotions[0] - sorted_emotions[1]) / sorted_emotions[0]
        else:
            confidence = 1.0 if sorted_emotions[0] > 50 else 0.5
        
        return min(1.0, confidence)
    
    def draw_emotion_bar(self, frame, emotion_data, position=(None, None)):
        """
        Dibuja una barra simple con la emoción dominante.
        
        Args:
            frame: Frame donde dibujar
            emotion_data: Datos de análisis emocional
            position: (x, y) posición de la barra
        """
        if self.headless or not emotion_data:
            return frame
        
        h, w = frame.shape[:2]
        
        # Posición por defecto
        bar_width = 250
        bar_height = 30
        
        if position[0] is None:
            bar_x = w - bar_width - 20
        else:
            bar_x = position[0]
            
        if position[1] is None:
            bar_y = 60  # Debajo de fatiga
        else:
            bar_y = position[1]
        
        # Emoción dominante
        emotion = emotion_data['dominant_emotion']
        percentage = emotion_data['emotions'][emotion]
        wellbeing = emotion_data['wellbeing']
        
        # Emoji para la emoción
        emotion_emojis = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'surprised': '😮', 'fear': '😨', 'disgust': '🤢',
            'neutral': '😐'
        }
        
        emoji = emotion_emojis.get(emotion, '😐')
        
        # Color basado en wellbeing
        if wellbeing > 70:
            color = (0, 255, 0)  # Verde
        elif wellbeing > 40:
            color = (0, 165, 255)  # Naranja
        else:
            color = (0, 0, 255)  # Rojo
        
        # Texto
        text = f"{emoji} {emotion.upper()}: {percentage}% (Wellbeing: {wellbeing}%)"
        cv2.putText(frame, text, (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Barra de wellbeing
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        progress_width = int(bar_width * (wellbeing / 100))
        if progress_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         color, -1)
        
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), 1)
        
        return frame
    
    def get_emotion_report(self):
        """
        Genera reporte JSON para el servidor.
        
        Returns:
            dict: Reporte de emociones
        """
        # Calcular estadísticas del historial
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(self.emotion_history) or 1
        emotion_distribution = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'timestamp': self.last_update,
            'current_emotion': self.emotion_history[-1] if self.emotion_history else 'neutral',
            'emotion_distribution': emotion_distribution,
            'stability': self._calculate_emotional_stability(),
            'is_calibrated': self.is_calibrated,
            'metrics': {
                'smile_ratio': self.last_metrics.get('smile_ratio', 0),
                'eye_opening': self.last_metrics.get('eye_opening_ratio', 0),
                'eyebrow_distance': self.last_metrics.get('eyebrow_distance_ratio', 0)
            }
        }
    
    def _calculate_emotional_stability(self):
        """Calcula estabilidad emocional"""
        if len(self.emotion_history) < 3:
            return 1.0
        
        changes = sum(1 for i in range(1, len(self.emotion_history)) 
                     if self.emotion_history[i] != self.emotion_history[i-1])
        
        stability = 1.0 - (changes / len(self.emotion_history))
        return round(stability, 2)
    
    # Métodos auxiliares existentes
    def _distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_center(self, points):
        """Calcula el centro de un conjunto de puntos"""
        if not points:
            return (0, 0)
        x = sum(p[0] for p in points) // len(points)
        y = sum(p[1] for p in points) // len(points)
        return (x, y)
    
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0.3
        
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        h = self._distance(eye_points[0], eye_points[3])
        
        return (v1 + v2) / (2.0 * h) if h > 0 else 0
    
    def _calculate_facial_asymmetry(self, landmarks):
        """Calcula asimetría facial básica"""
        # Comparar distancias de puntos simétricos
        left_eye = self._get_center(landmarks['left_eye'])
        right_eye = self._get_center(landmarks['right_eye'])
        nose_tip = landmarks['nose_tip'][2]  # Centro de la nariz
        
        # Distancia de cada ojo a la nariz
        left_dist = self._distance(left_eye, nose_tip)
        right_dist = self._distance(right_eye, nose_tip)
        
        # Asimetría normalizada
        if max(left_dist, right_dist) > 0:
            asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
        else:
            asymmetry = 0
        
        return asymmetry
    
    def _default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'emotions': {
                'happy': 0, 'sad': 0, 'angry': 0,
                'surprised': 0, 'fear': 0, 'disgust': 0,
                'neutral': 100
            },
            'dominant_emotion': 'neutral',
            'wellbeing': 50,
            'valence': 0,
            'arousal': 0,
            'confidence': 0,
            'is_calibrated': self.is_calibrated
        }
    
    def reset(self):
        """Reinicia el analizador"""
        self.emotion_history.clear()
        self.current_emotion = "neutral"
        self.last_metrics = {}