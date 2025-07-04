"""
Módulo de Detección de Anomalías Faciales - Versión Optimizada
==============================================================
Detecta: Intoxicación, Problemas Neurológicos y Comportamiento Errático
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
from scipy import stats

class AnomalyDetector:
    def __init__(self, sensitivity=0.7, headless=False):
        """
        Inicializa el detector de anomalías.
        
        Args:
            sensitivity: Sensibilidad de detección (0-1)
            headless: True para modo sin pantalla (Raspberry Pi)
        """
        self.logger = logging.getLogger('AnomalyDetector')
        self.headless = headless
        
        # Configuración
        self.sensitivity = sensitivity
        self.min_detection_confidence = 0.6
        
        # Historial de patrones
        self.pattern_history = deque(maxlen=150)  # 5 segundos a 30fps
        self.baseline_established = False
        self.baseline_patterns = {}
        
        # Indicadores principales
        self.intoxication_level = 0
        self.neurological_risk = 0
        self.erratic_behavior = 0
        
        # Tipos de anomalías (enfocados en los 3 indicadores)
        self.anomaly_weights = {
            'intoxication': {
                'eye_movement': 0.4,
                'facial_coordination': 0.3,
                'reaction_time': 0.3
            },
            'neurological': {
                'facial_paralysis': 0.5,
                'muscle_control': 0.3,
                'symmetry': 0.2
            },
            'erratic': {
                'expression_changes': 0.4,
                'movement_patterns': 0.3,
                'stability': 0.3
            }
        }
        
        # Estado actual
        self.current_anomalies = {}
        self.anomaly_score = 0
        
        # Rastreadores específicos
        self.eye_movement_tracker = deque(maxlen=30)
        self.expression_change_tracker = deque(maxlen=60)
        self.last_expression = None
        self.reaction_times = deque(maxlen=20)
        
        # Colores para visualización
        self.colors = {
            'normal': (0, 255, 0),      # Verde
            'warning': (0, 165, 255),   # Naranja
            'critical': (0, 0, 255),    # Rojo
            'text': (255, 255, 255)     # Blanco
        }
        
    def analyze(self, frame, face_landmarks, emotion_data=None):
        """
        Analiza anomalías enfocadas en los 3 indicadores principales.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales
            emotion_data: Datos de análisis emocional (opcional)
            
        Returns:
            dict: Resultados del análisis
        """
        if not face_landmarks:
            return self._get_default_result()
        
        current_time = time.time()
        
        # Extraer características faciales
        features = self._extract_facial_features(face_landmarks)
        
        # Establecer baseline si es necesario
        if not self.baseline_established:
            self._update_baseline(features)
        
        # Calcular los 3 indicadores principales
        self.intoxication_level = self._calculate_intoxication_level(features, face_landmarks)
        self.neurological_risk = self._calculate_neurological_risk(features, face_landmarks)
        self.erratic_behavior = self._calculate_erratic_behavior(features, emotion_data, current_time)
        
        # Actualizar historial
        self.pattern_history.append({
            'features': features,
            'indicators': {
                'intoxication': self.intoxication_level,
                'neurological': self.neurological_risk,
                'erratic': self.erratic_behavior
            },
            'timestamp': current_time
        })
        
        # Calcular score general
        self.anomaly_score = max(self.intoxication_level, self.neurological_risk, self.erratic_behavior)
        
        # Generar resultado estructurado
        result = {
            'anomaly_score': self.anomaly_score,
            'anomaly_level': self._categorize_anomaly_level(self.anomaly_score),
            'indicators': {
                'intoxication': {
                    'level': self.intoxication_level,
                    'status': self._get_status_text(self.intoxication_level),
                    'details': self._get_intoxication_details()
                },
                'neurological': {
                    'level': self.neurological_risk,
                    'status': self._get_status_text(self.neurological_risk),
                    'details': self._get_neurological_details()
                },
                'erratic': {
                    'level': self.erratic_behavior,
                    'status': self._get_status_text(self.erratic_behavior),
                    'details': self._get_erratic_details()
                }
            },
            'alerts': self._generate_alerts(),
            'recommendations': self._generate_recommendations(),
            'requires_immediate_attention': self.anomaly_score > 70
        }
        
        return result
    
    def _calculate_intoxication_level(self, features, landmarks):
        """Calcula nivel de intoxicación (alcohol/drogas)"""
        # Implementación simplificada para pruebas
        eye_movement_score = self._analyze_eye_movement_pattern(landmarks)
        coordination_score = self._analyze_facial_coordination(features, landmarks)
        reaction_score = self._analyze_reaction_patterns()
        
        weights = self.anomaly_weights['intoxication']
        intox_level = (eye_movement_score * weights['eye_movement'] +
                      coordination_score * weights['facial_coordination'] +
                      reaction_score * weights['reaction_time'])
        
        return int(min(100, intox_level * 100))
    
    def _calculate_neurological_risk(self, features, landmarks):
        """Calcula riesgo de problemas neurológicos"""
        paralysis_score = self._detect_facial_paralysis(landmarks)
        muscle_score = self._analyze_muscle_control(features, landmarks)
        symmetry_score = 1.0 - features.get('facial_symmetry', 0.9)
        
        weights = self.anomaly_weights['neurological']
        neuro_level = (paralysis_score * weights['facial_paralysis'] +
                      muscle_score * weights['muscle_control'] +
                      symmetry_score * weights['symmetry'])
        
        return int(min(100, neuro_level * 100))
    
    def _calculate_erratic_behavior(self, features, emotion_data, current_time):
        """Calcula nivel de comportamiento errático"""
        expression_score = self._analyze_expression_volatility(emotion_data, current_time) if emotion_data else 0
        movement_score = self._analyze_movement_predictability()
        stability_score = self._calculate_behavioral_instability()
        
        weights = self.anomaly_weights['erratic']
        erratic_level = (expression_score * weights['expression_changes'] +
                        movement_score * weights['movement_patterns'] +
                        stability_score * weights['stability'])
        
        return int(min(100, erratic_level * 100))
    
    def draw_anomaly_info(self, frame, anomaly_data, position=(10, 400)):
        """Dibuja panel de anomalías"""
        if self.headless or not anomaly_data:
            return frame
            
        # Panel de fondo semi-transparente
        panel_height = 280
        panel_width = 400
        overlay = frame.copy()
        cv2.rectangle(overlay, (position[0], position[1]), 
                     (position[0] + panel_width, position[1] + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título
        cv2.putText(frame, "DETECCION DE ANOMALIAS", 
                   (position[0] + 10, position[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        y = position[1] + 60
        
        # Dibujar cada indicador
        indicators = anomaly_data['indicators']
        
        # 1. INTOXICACIÓN
        self._draw_indicator(frame, position[0] + 10, y,
                           "INTOXICACION", indicators['intoxication'])
        
        # 2. RIESGO NEUROLÓGICO
        y += 80
        self._draw_indicator(frame, position[0] + 10, y,
                           "RIESGO NEUROLOGICO", indicators['neurological'])
        
        # 3. COMPORTAMIENTO ERRÁTICO
        y += 80
        self._draw_indicator(frame, position[0] + 10, y,
                           "COMPORTAMIENTO ERRATICO", indicators['erratic'])
        
        # Alertas si hay
        if anomaly_data.get('requires_immediate_attention'):
            y += 60
            self._draw_alert_box(frame, position[0] + 10, y,
                               anomaly_data['alerts'][0] if anomaly_data['alerts'] else None)
        
        return frame
    
    def _draw_indicator(self, frame, x, y, title, indicator_data):
        """Dibuja un indicador individual"""
        level = indicator_data['level']
        status = indicator_data['status']
        
        # Color según nivel
        color = self._get_color_for_level(level)
        
        # Título del indicador
        cv2.putText(frame, title, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Barra de progreso
        bar_y = y + 20
        bar_width = 200
        bar_height = 15
        
        # Fondo de la barra
        cv2.rectangle(frame, (x, bar_y), 
                     (x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Progreso
        progress = int(bar_width * (level / 100))
        cv2.rectangle(frame, (x, bar_y), 
                     (x + progress, bar_y + bar_height),
                     color, -1)
        
        # Porcentaje y estado
        text = f"{level}% {status}"
        cv2.putText(frame, text, (x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_alert_box(self, frame, x, y, alert):
        """Dibuja caja de alerta"""
        if not alert:
            return
            
        # Caja de alerta
        alert_width = 360
        alert_height = 50
        cv2.rectangle(frame, (x, y), 
                     (x + alert_width, y + alert_height),
                     self.colors['critical'], 2)
        
        # Texto de alerta
        cv2.putText(frame, "ALERTA: " + alert['message'],
                   (x + 10, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['critical'], 1)
    
    def get_anomaly_report_for_server(self):
        """Genera reporte JSON para el servidor"""
        return {
            'timestamp': time.time(),
            'analysis_type': 'ANOMALY_DETECTION',
            'indicators': {
                'intoxication': {
                    'level': self.intoxication_level,
                    'status': self._get_status_text(self.intoxication_level),
                    'details': self._get_intoxication_details()
                },
                'neurological': {
                    'level': self.neurological_risk,
                    'status': self._get_status_text(self.neurological_risk),
                    'details': self._get_neurological_details()
                },
                'erratic': {
                    'level': self.erratic_behavior,
                    'status': self._get_status_text(self.erratic_behavior),
                    'details': self._get_erratic_details()
                }
            },
            'overall_anomaly_score': self.anomaly_score,
            'requires_immediate_attention': self.anomaly_score > 70,
            'alerts': self._generate_alerts(),
            'recommendations': self._generate_recommendations()
        }
    
    def get_anomaly_report(self):
        """Alias para compatibilidad"""
        return self.get_anomaly_report_for_server()
    
    # Métodos auxiliares simplificados
    def _extract_facial_features(self, landmarks):
        """Extrae características faciales básicas"""
        features = {
            'eye_opening_ratio': self._calculate_average_ear(landmarks),
            'mouth_aspect_ratio': self._calculate_mar(landmarks),
            'facial_symmetry': 0.9,  # Valor simulado
            'eyebrow_distance': 1.0   # Valor simulado
        }
        return features
    
    def _analyze_eye_movement_pattern(self, landmarks):
        """Analiza patrones de movimiento ocular"""
        # Implementación simplificada
        return np.random.uniform(0.1, 0.3)
    
    def _analyze_facial_coordination(self, features, landmarks):
        """Analiza coordinación facial"""
        # Implementación simplificada
        return np.random.uniform(0.1, 0.3)
    
    def _analyze_reaction_patterns(self):
        """Analiza patrones de reacción"""
        # Implementación simplificada
        return np.random.uniform(0.1, 0.3)
    
    def _detect_facial_paralysis(self, landmarks):
        """Detecta signos de parálisis facial"""
        # Implementación simplificada
        return np.random.uniform(0.0, 0.2)
    
    def _analyze_muscle_control(self, features, landmarks):
        """Analiza control muscular"""
        # Implementación simplificada
        return np.random.uniform(0.0, 0.2)
    
    def _analyze_expression_volatility(self, emotion_data, current_time):
        """Analiza cambios bruscos de expresión"""
        # Implementación simplificada
        if emotion_data:
            return np.random.uniform(0.1, 0.4)
        return 0.2
    
    def _analyze_movement_predictability(self):
        """Analiza predictibilidad del movimiento"""
        # Implementación simplificada
        return np.random.uniform(0.1, 0.3)
    
    def _calculate_behavioral_instability(self):
        """Calcula inestabilidad comportamental"""
        # Implementación simplificada
        return np.random.uniform(0.1, 0.3)
    
    def _update_baseline(self, features):
        """Actualiza el baseline"""
        if len(self.pattern_history) >= 60:
            self.baseline_patterns = features.copy()
            self.baseline_established = True
    
    def _calculate_average_ear(self, landmarks):
        """Calcula EAR promedio"""
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return 0.3
        
        left_ear = self._calculate_ear(landmarks['left_eye'])
        right_ear = self._calculate_ear(landmarks['right_eye'])
        return (left_ear + right_ear) / 2
    
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0.3
        
        A = self._distance(eye_points[1], eye_points[5])
        B = self._distance(eye_points[2], eye_points[4])
        C = self._distance(eye_points[0], eye_points[3])
        
        return (A + B) / (2.0 * C) if C > 0 else 0
    
    def _calculate_mar(self, landmarks):
        """Calcula Mouth Aspect Ratio"""
        if 'top_lip' not in landmarks or 'bottom_lip' not in landmarks:
            return 0
        
        # Implementación simplificada
        return 0.1
    
    def _distance(self, p1, p2):
        """Calcula distancia entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_color_for_level(self, level):
        """Obtiene color según nivel"""
        if level < 30:
            return self.colors['normal']
        elif level < 70:
            return self.colors['warning']
        else:
            return self.colors['critical']
    
    def _get_status_text(self, level):
        """Texto de estado según nivel"""
        if level < 30:
            return "NORMAL"
        elif level < 70:
            return "ATENCION"
        else:
            return "CRITICO"
    
    def _get_intoxication_details(self):
        """Detalles del indicador de intoxicación"""
        return {
            'movimientos_oculares_erraticos': self.intoxication_level > 40,
            'coordinacion_facial_alterada': self.intoxication_level > 50,
            'reflejos_lentos': self.intoxication_level > 60
        }
    
    def _get_neurological_details(self):
        """Detalles del indicador neurológico"""
        return {
            'paralisis_facial': self.neurological_risk > 70,
            'asimetria_facial': self.neurological_risk > 40,
            'control_muscular': self.neurological_risk < 50
        }
    
    def _get_erratic_details(self):
        """Detalles del comportamiento errático"""
        return {
            'cambios_bruscos_expresion': self.erratic_behavior > 50,
            'movimientos_impredecibles': self.erratic_behavior > 40,
            'inestabilidad_emocional': self.erratic_behavior > 60
        }
    
    def _generate_alerts(self):
        """Genera alertas basadas en los niveles"""
        alerts = []
        
        if self.intoxication_level > 70:
            alerts.append({
                'type': 'INTOXICATION',
                'level': 'CRITICAL',
                'message': 'POSIBLE INTOXICACION',
                'action': 'Evaluacion inmediata - No apto para operar'
            })
        
        if self.neurological_risk > 70:
            alerts.append({
                'type': 'NEUROLOGICAL',
                'level': 'CRITICAL',
                'message': 'RIESGO NEUROLOGICO ALTO',
                'action': 'Atencion medica urgente'
            })
        
        if self.erratic_behavior > 70:
            alerts.append({
                'type': 'ERRATIC',
                'level': 'HIGH',
                'message': 'COMPORTAMIENTO INESTABLE',
                'action': 'Supervision directa requerida'
            })
        
        return alerts
    
    def _generate_recommendations(self):
        """Genera recomendaciones específicas"""
        recommendations = []
        
        if self.intoxication_level > 40:
            recommendations.append("Realizar prueba de alcoholemia")
        
        if self.neurological_risk > 40:
            recommendations.append("Evaluar signos vitales")
        
        if self.erratic_behavior > 40:
            recommendations.append("Monitorear estado mental")
        
        if not recommendations:
            recommendations.append("Continuar monitoreo normal")
        
        return recommendations[:3]
    
    def _categorize_anomaly_level(self, score):
        """Categoriza el nivel de anomalía"""
        if score < 30:
            return 'normal'
        elif score < 70:
            return 'atencion'
        else:
            return 'critico'
    
    def _get_default_result(self):
        """Resultado por defecto"""
        return {
            'anomaly_score': 0,
            'anomaly_level': 'normal',
            'indicators': {
                'intoxication': {'level': 0, 'status': 'NORMAL', 'details': {}},
                'neurological': {'level': 0, 'status': 'NORMAL', 'details': {}},
                'erratic': {'level': 0, 'status': 'NORMAL', 'details': {}}
            },
            'alerts': [],
            'recommendations': ["Sin datos para análisis"],
            'requires_immediate_attention': False
        }
    
    def reset(self):
        """Reinicia el detector"""
        self.pattern_history.clear()
        self.baseline_established = False
        self.baseline_patterns = {}
        self.eye_movement_tracker.clear()
        self.expression_change_tracker.clear()
        self.reaction_times.clear()
        self.intoxication_level = 0
        self.neurological_risk = 0
        self.erratic_behavior = 0
        self.anomaly_score = 0