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
        
        # Modo calibración
        self.calibration_mode = False
        self.calibration_data = {
            'neutral': [],
            'smile': [],
            'frown': []
        }
        self.calibrated_thresholds = None
        
        # Colores para cada emoción (BGR)
        self.emotion_colors = {
            'neutral': (200, 200, 200),   # Gris
            'alegria': (0, 255, 0),       # Verde
            'tristeza': (255, 0, 0),      # Azul
            'enojo': (0, 0, 255),         # Rojo
            'frustracion': (0, 100, 255), # Rojo oscuro
            'concentracion': (255, 165, 0), # Naranja
            'relajacion': (0, 255, 255),  # Amarillo
            'desanimo': (128, 0, 128)     # Púrpura
        }
        
        # Métricas de la última detección
        self.last_metrics = {}
        
    def calibrate(self, expression_type, metrics):
        """
        Calibra el analizador con expresiones específicas del usuario.
        
        Args:
            expression_type: 'neutral', 'smile', o 'frown'
            metrics: Métricas faciales actuales
        """
        if expression_type in self.calibration_data:
            self.calibration_data[expression_type].append(metrics.copy())
            self.logger.info(f"Calibración: {expression_type} - muestra {len(self.calibration_data[expression_type])}")
            
            # Si tenemos suficientes muestras, calcular umbrales
            if all(len(data) >= 10 for data in self.calibration_data.values()):
                self._calculate_calibrated_thresholds()
                
    def _calculate_calibrated_thresholds(self):
        """Calcula umbrales personalizados basados en calibración"""
        self.calibrated_thresholds = {}
        
        # Calcular promedios para cada expresión
        neutral_avg = self._average_metrics(self.calibration_data['neutral'])
        smile_avg = self._average_metrics(self.calibration_data['smile'])
        frown_avg = self._average_metrics(self.calibration_data['frown'])
        
        # Establecer umbrales adaptativos
        self.calibrated_thresholds['smile_threshold'] = (neutral_avg['smile_ratio'] + smile_avg['smile_ratio']) / 2
        self.calibrated_thresholds['frown_threshold'] = (neutral_avg['mouth_corner_drop'] + frown_avg['mouth_corner_drop']) / 2
        self.calibrated_thresholds['eyebrow_threshold'] = (neutral_avg['eyebrow_distance_ratio'] + frown_avg['eyebrow_distance_ratio']) / 2
        
        self.logger.info(f"Calibración completada: {self.calibrated_thresholds}")
        self.calibration_mode = False
        
    def _average_metrics(self, metrics_list):
        """Calcula el promedio de una lista de métricas"""
        if not metrics_list:
            return {}
        
        avg = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg[key] = np.mean(values) if values else 0
        
        return avg
        
    def analyze(self, frame, face_landmarks):
        """
        Analiza la expresión facial basada en landmarks.
        
        Args:
            frame: Imagen del rostro
            face_landmarks: Diccionario con puntos faciales
            
        Returns:
            dict: {
                'emotions': dict con porcentajes de cada emoción,
                'dominant_emotion': str,
                'wellbeing': float (0-100),
                'stress_level': float (0-100),
                'stability': str ('ALTA', 'MEDIA', 'BAJA'),
                'anomalies': float (0-100),
                'metrics': dict con métricas detalladas
            }
        """
        if not face_landmarks:
            return self._default_result()
        
        try:
            # Calcular métricas faciales
            metrics = self._calculate_facial_metrics(face_landmarks)
            self.last_metrics = metrics
            
            # Clasificar emociones
            emotions = self._classify_emotions(metrics)
            
            # Calcular métricas derivadas
            wellbeing = self._calculate_wellbeing(emotions)
            stress_level = self._calculate_stress_level(emotions, metrics)
            stability = self._calculate_emotional_stability()
            anomalies = self._detect_anomalies(metrics)
            
            # Determinar emoción dominante
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Agregar al historial
            self.emotion_history.append(dominant_emotion)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'wellbeing': wellbeing,
                'stress_level': stress_level,
                'stability': stability,
                'anomalies': anomalies,
                'metrics': metrics
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
        
        # Calcular curvatura - MEJORADO
        mouth_width = self._distance(mouth_left, mouth_right)
        
        # Usar punto medio del labio superior para mejor detección
        upper_lip_center = self._midpoint(landmarks['top_lip'][2], landmarks['top_lip'][4])
        
        # Calcular diferencia vertical entre centro y esquinas
        left_corner_height = mouth_left[1]
        right_corner_height = mouth_right[1]
        center_height = upper_lip_center[1]
        
        # Promedio de elevación de comisuras (negativo = sonrisa)
        corner_elevation = ((left_corner_height + right_corner_height) / 2) - center_height
        
        # Normalizar por tamaño de boca
        metrics['smile_ratio'] = abs(corner_elevation) / mouth_width if mouth_width > 0 else 0
        
        # Si las comisuras están más arriba que el centro, es sonrisa
        if corner_elevation > 0:
            metrics['smile_ratio'] *= -1  # Hacer negativo para indicar comisuras caídas
        
        # 2. Apertura de boca - MEJORADO
        # Usar múltiples puntos para mayor precisión
        top_lip_points = [landmarks['top_lip'][2], landmarks['top_lip'][3], landmarks['top_lip'][4]]
        bottom_lip_points = [landmarks['bottom_lip'][2], landmarks['bottom_lip'][3], landmarks['bottom_lip'][4]]
        
        # Calcular distancia promedio
        mouth_distances = []
        for t, b in zip(top_lip_points, bottom_lip_points):
            mouth_distances.append(self._distance(t, b))
        
        mouth_opening = np.mean(mouth_distances)
        metrics['mouth_opening_ratio'] = mouth_opening / mouth_width if mouth_width > 0 else 0
        
        # 3. Elevación de cejas - MEJORADO
        # Usar punto medio de cada ceja
        left_eyebrow_center = self._midpoint(landmarks['left_eyebrow'][1], landmarks['left_eyebrow'][3])
        right_eyebrow_center = self._midpoint(landmarks['right_eyebrow'][1], landmarks['right_eyebrow'][3])
        
        # Punto medio de cada ojo
        left_eye_center = self._get_center(landmarks['left_eye'])
        right_eye_center = self._get_center(landmarks['right_eye'])
        
        # Distancia vertical entre ceja y ojo
        left_eyebrow_height = abs(left_eyebrow_center[1] - left_eye_center[1])
        right_eyebrow_height = abs(right_eyebrow_center[1] - right_eye_center[1])
        
        eyebrow_eye_distance = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # Normalizar por altura facial
        face_height = self._distance(landmarks['chin'][8], landmarks['left_eyebrow'][2])
        metrics['eyebrow_raise_ratio'] = eyebrow_eye_distance / face_height if face_height > 0 else 0
        
        # 4. Fruncimiento de ceño (cejas juntas) - MEJORADO
        # Puntos interiores de las cejas
        left_inner = landmarks['left_eyebrow'][4]
        right_inner = landmarks['right_eyebrow'][0]
        
        eyebrow_distance = self._distance(left_inner, right_inner)
        
        # Normalizar por distancia entre ojos (más estable que ancho de boca)
        eye_distance = self._distance(left_eye_center, right_eye_center)
        metrics['eyebrow_distance_ratio'] = eyebrow_distance / eye_distance if eye_distance > 0 else 1.0
        
        # 5. Comisuras de la boca (tristeza) - MEJORADO
        # Usar labio inferior también para mejor detección
        mouth_corner_left = self._midpoint(landmarks['top_lip'][0], landmarks['bottom_lip'][0])
        mouth_corner_right = self._midpoint(landmarks['top_lip'][6], landmarks['bottom_lip'][6])
        mouth_center_full = self._midpoint(upper_lip_center, 
                                          self._midpoint(landmarks['bottom_lip'][3], landmarks['bottom_lip'][4]))
        
        # Calcular caída de comisuras
        corner_drop_left = mouth_corner_left[1] - mouth_center_full[1]
        corner_drop_right = mouth_corner_right[1] - mouth_center_full[1]
        corner_drop = (corner_drop_left + corner_drop_right) / 2
        
        # Normalizar
        metrics['mouth_corner_drop'] = corner_drop / face_height if face_height > 0 else 0
        
        # 6. Apertura de ojos - MEJORADO
        left_eye_opening = self._calculate_eye_opening(landmarks['left_eye'])
        right_eye_opening = self._calculate_eye_opening(landmarks['right_eye'])
        metrics['eye_opening_ratio'] = (left_eye_opening + right_eye_opening) / 2
        
        # 7. Simetría facial
        left_face_points = self._get_left_face_points(landmarks)
        right_face_points = self._get_right_face_points(landmarks)
        metrics['facial_symmetry'] = self._calculate_symmetry(left_face_points, right_face_points)
        
        # 8. Tensión facial general
        metrics['facial_tension'] = self._calculate_facial_tension(landmarks)
        
        return metrics
    
    def _classify_emotions(self, metrics):
        """Clasifica las emociones basadas en las métricas"""
        emotions = {
            'alegria': 0,
            'tristeza': 0,
            'enojo': 0,
            'frustracion': 0,
            'concentracion': 0,
            'relajacion': 0,
            'desanimo': 0,
            'neutral': 40  # Base neutral aumentada
        }
        
        # Debug: imprimir métricas para calibración
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 30 == 0:  # Cada segundo aprox
            self.logger.info(f"Métricas: smile={metrics['smile_ratio']:.3f}, "
                           f"mouth_drop={metrics['mouth_corner_drop']:.3f}, "
                           f"eyebrow_dist={metrics['eyebrow_distance_ratio']:.3f}, "
                           f"eye_open={metrics['eye_opening_ratio']:.3f}")
        
        # ALEGRÍA - Ajustado para ser más sensible
        if metrics['smile_ratio'] > 0.015:  # Umbral reducido
            intensity = min(1.0, metrics['smile_ratio'] / 0.06)  # Normalizar
            emotions['alegria'] = int(20 + intensity * 60)  # 20-80%
            emotions['neutral'] -= int(intensity * 20)
            
            # Bonus si las comisuras suben
            if metrics['mouth_corner_drop'] < -0.005:
                emotions['alegria'] = min(emotions['alegria'] + 10, 90)
        
        # TRISTEZA - Más sensible a comisuras caídas
        if metrics['mouth_corner_drop'] > 0.01:  # Umbral reducido
            intensity = min(1.0, metrics['mouth_corner_drop'] / 0.04)
            emotions['tristeza'] = int(15 + intensity * 55)
            emotions['neutral'] -= int(intensity * 15)
            
            # Bonus si ojos más cerrados
            if metrics['eye_opening_ratio'] < 0.23:
                emotions['tristeza'] = min(emotions['tristeza'] + 10, 80)
        
        # ENOJO - Basado en cejas juntas y tensión
        if metrics['eyebrow_distance_ratio'] < 0.92:  # Ajustado
            intensity = min(1.0, (0.92 - metrics['eyebrow_distance_ratio']) / 0.15)
            emotions['enojo'] = int(20 + intensity * 55)
            emotions['neutral'] -= int(intensity * 20)
            
            # Bonus si hay tensión facial
            if metrics.get('facial_tension', 0) > 0.5:
                emotions['enojo'] = min(emotions['enojo'] + 15, 85)
        
        # FRUSTRACIÓN - Combinación de indicadores
        tension = metrics.get('facial_tension', 0)
        if tension > 0.4 and metrics['eyebrow_distance_ratio'] < 0.95:
            emotions['frustracion'] = int(tension * 60)
            emotions['neutral'] -= 10
        
        # CONCENTRACIÓN - Ojos ligeramente entrecerrados + cejas
        if (0.18 < metrics['eye_opening_ratio'] < 0.26 and 
            metrics['eyebrow_distance_ratio'] < 0.94):
            emotions['concentracion'] = 40
            emotions['neutral'] -= 15
        
        # RELAJACIÓN - Rostro relajado
        if (tension < 0.25 and 
            0.24 < metrics['eye_opening_ratio'] < 0.30 and 
            abs(metrics['mouth_corner_drop']) < 0.008):
            emotions['relajacion'] = int((1 - tension) * 60)
            emotions['neutral'] -= 10
        
        # DESÁNIMO - Combinación de tristeza y baja energía
        if (metrics['mouth_corner_drop'] > 0.008 and 
            metrics['eyebrow_raise_ratio'] < 0.09 and 
            metrics['eye_opening_ratio'] < 0.24):
            emotions['desanimo'] = 45
            emotions['neutral'] -= 15
        
        # Asegurar que neutral no sea negativo
        emotions['neutral'] = max(10, emotions['neutral'])
        
        # Normalizar a 100%
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] = int((emotions[emotion] / total) * 100)
        
        return emotions
    
    def _calculate_wellbeing(self, emotions):
        """Calcula el nivel de bienestar basado en emociones positivas vs negativas"""
        positive = emotions['alegria'] + emotions['relajacion'] + emotions['neutral'] * 0.5
        negative = emotions['tristeza'] + emotions['enojo'] + emotions['frustracion'] + emotions['desanimo']
        
        total = positive + negative
        if total > 0:
            wellbeing = (positive / total) * 100
        else:
            wellbeing = 50  # Neutral
        
        return int(wellbeing)
    
    def _calculate_stress_level(self, emotions, metrics):
        """Calcula el nivel de estrés combinando emociones y tensión facial"""
        # Componente emocional
        stress_emotions = emotions['enojo'] + emotions['frustracion'] + emotions['desanimo'] * 0.5
        
        # Componente físico
        physical_stress = metrics['facial_tension'] * 100
        
        # Combinar ambos
        stress_level = (stress_emotions * 0.6 + physical_stress * 0.4)
        
        return int(min(100, stress_level))
    
    def _calculate_emotional_stability(self):
        """Calcula la estabilidad emocional basada en el historial"""
        if len(self.emotion_history) < 5:
            return "ALTA"
        
        # Contar cambios de emoción
        changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i] != self.emotion_history[i-1]:
                changes += 1
        
        change_rate = changes / len(self.emotion_history)
        
        if change_rate < 0.3:
            return "ALTA"
        elif change_rate < 0.6:
            return "MEDIA"
        else:
            return "BAJA"
    
    def _detect_anomalies(self, metrics):
        """Detecta comportamientos faciales anómalos"""
        anomaly_score = 0
        
        # Asimetría facial extrema
        if metrics['facial_symmetry'] < 0.7:
            anomaly_score += 30
        
        # Tensión facial muy alta
        if metrics['facial_tension'] > 0.8:
            anomaly_score += 25
        
        # Expresiones contradictorias (ej: sonrisa con cejas fruncidas)
        if metrics['smile_ratio'] > 0.02 and metrics['eyebrow_distance_ratio'] < 0.85:
            anomaly_score += 20
        
        # Apertura de ojos anormal
        if metrics['eye_opening_ratio'] < 0.15 or metrics['eye_opening_ratio'] > 0.35:
            anomaly_score += 15
        
        # Movimientos de boca inusuales
        if metrics['mouth_opening_ratio'] > 0.3:
            anomaly_score += 10
        
        return min(100, anomaly_score)
    
    def draw_emotion_panel(self, frame, analysis_result, position=(None, 50)):
        """Dibuja el panel de análisis emocional en el lado derecho del frame"""
        if not analysis_result:
            return frame
        
        h, w = frame.shape[:2]
        panel_width = 300
        panel_x = w - panel_width - 10 if position[0] is None else position[0]
        panel_y = position[1]
        
        # Crear overlay semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w-10, h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título del panel
        cv2.putText(frame, "=== ANALISIS EMOCIONAL ===", 
                   (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dibujar emociones con barras
        y_offset = panel_y + 60
        emotions = analysis_result['emotions']
        
        # Mostrar TODAS las emociones, ordenadas por porcentaje
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, percentage in sorted_emotions:
            # Solo mostrar si tiene más del 5%
            if percentage > 5:
                # Nombre de emoción con emoji
                emotion_emojis = {
                    'alegria': '😊',
                    'enojo': '😠',
                    'tristeza': '😢',
                    'neutral': '😐',
                    'frustracion': '😤',
                    'concentracion': '🤔',
                    'relajacion': '😌',
                    'desanimo': '😔'
                }
                
                emoji = emotion_emojis.get(emotion, '')
                text = f"{emoji} {emotion.capitalize()}: {percentage}%"
                
                # Color de la emoción
                text_color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Si es la emoción dominante, hacerla más brillante
                if emotion == analysis_result['dominant_emotion']:
                    cv2.putText(frame, text, (panel_x + 20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                else:
                    cv2.putText(frame, text, (panel_x + 20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Barra de progreso
                bar_x = panel_x + 150
                bar_width = 100
                bar_height = 10
                
                # Fondo de la barra
                cv2.rectangle(frame, (bar_x, y_offset - 8), 
                             (bar_x + bar_width, y_offset - 8 + bar_height),
                             (50, 50, 50), -1)
                
                # Progreso
                progress_width = int(bar_width * (percentage / 100))
                cv2.rectangle(frame, (bar_x, y_offset - 8),
                             (bar_x + progress_width, y_offset - 8 + bar_height),
                             text_color, -1)
                
                y_offset += 25
        
        # Separador
        cv2.line(frame, (panel_x + 20, y_offset), (w - 30, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        # Indicadores de bienestar y estrés
        wellbeing = analysis_result['wellbeing']
        stress = analysis_result['stress_level']
        
        # Bienestar
        wellbeing_color = (0, 255, 0) if wellbeing > 70 else (0, 165, 255) if wellbeing > 40 else (0, 0, 255)
        cv2.putText(frame, f"💚 Bienestar: {wellbeing}%", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, wellbeing_color, 2)
        y_offset += 30
        
        # Estrés
        stress_color = (0, 255, 0) if stress < 30 else (0, 165, 255) if stress < 60 else (0, 0, 255)
        cv2.putText(frame, f"😰 Estres: {stress}%", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_color, 2)
        y_offset += 30
        
        # Estabilidad
        stability = analysis_result['stability']
        stability_color = (0, 255, 0) if stability == "ALTA" else (0, 165, 255) if stability == "MEDIA" else (0, 0, 255)
        cv2.putText(frame, f"🔄 Estabilidad: {stability}", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        y_offset += 40
        
        # Separador
        cv2.line(frame, (panel_x + 20, y_offset), (w - 30, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        # Anomalías
        anomalies = analysis_result['anomalies']
        anomaly_color = (0, 255, 0) if anomalies < 20 else (0, 165, 255) if anomalies < 50 else (0, 0, 255)
        cv2.putText(frame, f"⚠️  Anomalias: {anomalies}%", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, anomaly_color, 1)
        y_offset += 30
        
        # Métricas debug (opcional)
        if hasattr(self, 'last_metrics') and self.last_metrics:
            cv2.putText(frame, "--- Debug Metricas ---", 
                       (panel_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += 20
            
            debug_metrics = [
                ('Sonrisa', self.last_metrics.get('smile_ratio', 0)),
                ('Comisuras', self.last_metrics.get('mouth_corner_drop', 0)),
                ('Cejas', self.last_metrics.get('eyebrow_distance_ratio', 0)),
                ('Ojos', self.last_metrics.get('eye_opening_ratio', 0))
            ]
            
            for metric_name, value in debug_metrics:
                text = f"{metric_name}: {value:.3f}"
                cv2.putText(frame, text, (panel_x + 25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                y_offset += 15
        
        y_offset += 10
        
        # Estado general
        cv2.line(frame, (panel_x + 20, y_offset), (w - 30, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        cv2.putText(frame, "📊 Estado general:", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Determinar estado general
        if wellbeing > 70 and stress < 30 and stability == "ALTA":
            general_state = "EXCELENTE"
            state_color = (0, 255, 0)
        elif wellbeing > 50 and stress < 50:
            general_state = "EQUILIBRADO"
            state_color = (0, 255, 255)
        elif stress > 70 or wellbeing < 30:
            general_state = "REQUIERE ATENCION"
            state_color = (0, 0, 255)
        else:
            general_state = "REGULAR"
            state_color = (0, 165, 255)
        
        cv2.putText(frame, f"    {general_state}", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        return frame
    
    # Métodos auxiliares
    def _distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _midpoint(self, p1, p2):
        """Calcula punto medio entre dos puntos"""
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    
    def _get_center(self, points):
        """Calcula el centro de un conjunto de puntos"""
        if not points:
            return (0, 0)
        x = sum(p[0] for p in points) // len(points)
        y = sum(p[1] for p in points) // len(points)
        return (x, y)
    
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
    
    def _get_left_face_points(self, landmarks):
        """Obtiene puntos del lado izquierdo de la cara"""
        points = []
        # Tomar puntos relevantes del lado izquierdo
        points.extend(landmarks['left_eyebrow'])
        points.extend(landmarks['left_eye'])
        points.extend(landmarks['chin'][:9])  # Mitad izquierda del mentón
        return points
    
    def _get_right_face_points(self, landmarks):
        """Obtiene puntos del lado derecho de la cara"""
        points = []
        # Tomar puntos relevantes del lado derecho
        points.extend(landmarks['right_eyebrow'])
        points.extend(landmarks['right_eye'])
        points.extend(landmarks['chin'][8:])  # Mitad derecha del mentón
        return points
    
    def _calculate_symmetry(self, left_points, right_points):
        """Calcula simetría facial comparando lados"""
        if not left_points or not right_points:
            return 1.0
        
        # Calcular centro de la cara
        all_points = left_points + right_points
        center_x = sum(p[0] for p in all_points) / len(all_points)
        
        # Comparar distancias desde el centro
        left_distances = [abs(p[0] - center_x) for p in left_points]
        right_distances = [abs(p[0] - center_x) for p in right_points]
        
        # Calcular diferencia promedio
        min_len = min(len(left_distances), len(right_distances))
        diff_sum = sum(abs(left_distances[i] - right_distances[i]) 
                      for i in range(min_len))
        
        avg_diff = diff_sum / min_len if min_len > 0 else 0
        
        # Normalizar a 0-1 (1 = perfectamente simétrico)
        symmetry = 1.0 - min(1.0, avg_diff / 50)
        
        return symmetry
    
    def _calculate_facial_tension(self, landmarks):
        """Calcula tensión facial general"""
        tension_indicators = []
        
        # Tensión en las cejas
        left_brow = landmarks['left_eyebrow']
        right_brow = landmarks['right_eyebrow']
        
        # Distancia entre cejas (menor = más tensión)
        brow_dist = self._distance(left_brow[-1], right_brow[0])
        face_width = self._distance(landmarks['chin'][0], landmarks['chin'][16])
        
        if face_width > 0:
            normalized_brow_dist = brow_dist / face_width
            brow_tension = 1.0 - min(1.0, normalized_brow_dist)
            tension_indicators.append(brow_tension)
        
        # Tensión en la mandíbula (variación en el contorno)
        jaw_points = landmarks['chin']
        jaw_variations = []
        for i in range(1, len(jaw_points)-1):
            angle = self._calculate_angle(jaw_points[i-1], jaw_points[i], jaw_points[i+1])
            jaw_variations.append(abs(180 - angle))
        
        avg_jaw_variation = sum(jaw_variations) / len(jaw_variations) if jaw_variations else 0
        jaw_tension = min(1.0, avg_jaw_variation / 30)
        tension_indicators.append(jaw_tension)
        
        # Tensión alrededor de los ojos
        left_eye_area = self._calculate_area(landmarks['left_eye'])
        right_eye_area = self._calculate_area(landmarks['right_eye'])
        
        # Ojos más pequeños = más tensión
        avg_eye_area = (left_eye_area + right_eye_area) / 2
        normalized_eye_area = avg_eye_area / (face_width * face_width) if face_width > 0 else 0.1
        eye_tension = 1.0 - min(1.0, normalized_eye_area * 50)
        tension_indicators.append(eye_tension)
        
        # Promedio de todos los indicadores
        overall_tension = sum(tension_indicators) / len(tension_indicators) if tension_indicators else 0
        
        return overall_tension
    
    def _calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo formado por tres puntos"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_area(self, points):
        """Calcula el área de un polígono definido por puntos"""
        if len(points) < 3:
            return 0
        
        # Fórmula del área de un polígono
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def _default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'emotions': {
                'alegria': 0,
                'tristeza': 0,
                'enojo': 0,
                'frustracion': 0,
                'concentracion': 0,
                'relajacion': 0,
                'desanimo': 0,
                'neutral': 100
            },
            'dominant_emotion': 'neutral',
            'wellbeing': 50,
            'stress_level': 0,
            'stability': 'ALTA',
            'anomalies': 0,
            'metrics': {}
        }
    
    def get_emotion_summary(self, time_window=60):
        """Obtiene un resumen de emociones en la ventana de tiempo"""
        # Esta función podría expandirse para análisis temporal más complejo
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
    
    def reset(self):
        """Reinicia el analizador"""
        self.emotion_history.clear()
        self.current_emotion = "neutral"
        self.last_metrics = {}