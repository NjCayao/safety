"""
Módulo de Detección de Anomalías Faciales
=========================================
Detecta comportamientos faciales atípicos y anomalías.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
from scipy import stats

class AnomalyDetector:
    def __init__(self, sensitivity=0.7):
        """
        Inicializa el detector de anomalías.
        
        Args:
            sensitivity: Sensibilidad de detección (0-1)
        """
        self.logger = logging.getLogger('AnomalyDetector')
        
        # Configuración
        self.sensitivity = sensitivity
        self.min_detection_confidence = 0.6
        
        # Historial de patrones
        self.pattern_history = deque(maxlen=150)  # 5 segundos a 30fps
        self.baseline_established = False
        self.baseline_patterns = {}
        
        # Tipos de anomalías
        self.anomaly_types = {
            'asymmetry': {'weight': 0.25, 'threshold': 0.3},
            'unnatural_expression': {'weight': 0.20, 'threshold': 0.5},
            'eye_movement': {'weight': 0.20, 'threshold': 0.4},
            'prolonged_smile': {'weight': 0.15, 'threshold': 0.6},
            'facial_rigidity': {'weight': 0.10, 'threshold': 0.7},
            'micro_expressions': {'weight': 0.10, 'threshold': 0.5}
        }
        
        # Estado actual
        self.current_anomalies = {}
        self.anomaly_score = 0
        
        # Rastreadores específicos
        self.smile_tracker = {'start_time': None, 'duration': 0}
        self.eye_movement_tracker = deque(maxlen=30)
        self.expression_change_tracker = deque(maxlen=60)
        self.last_expression = None
        
        # Colores para visualización
        self.anomaly_colors = {
            'low': (0, 255, 0),      # Verde
            'medium': (0, 165, 255),  # Naranja
            'high': (0, 0, 255)       # Rojo
        }
        
    def analyze(self, frame, face_landmarks, emotion_data=None):
        """
        Analiza anomalías en el comportamiento facial.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales
            emotion_data: Datos de análisis emocional (opcional)
            
        Returns:
            dict: Resultados del análisis de anomalías
        """
        if not face_landmarks:
            return self._get_default_result()
        
        current_time = time.time()
        
        # Extraer características faciales
        features = self._extract_facial_features(face_landmarks)
        
        # Establecer baseline si es necesario
        if not self.baseline_established:
            self._update_baseline(features)
        
        # Detectar diferentes tipos de anomalías
        anomalies = {
            'asymmetry': self._detect_facial_asymmetry(face_landmarks),
            'unnatural_expression': self._detect_unnatural_expression(features, emotion_data),
            'eye_movement': self._detect_irregular_eye_movement(face_landmarks),
            'prolonged_smile': self._detect_prolonged_smile(features, emotion_data, current_time),
            'facial_rigidity': self._detect_facial_rigidity(features),
            'micro_expressions': self._detect_micro_expressions(emotion_data)
        }
        
        # Calcular score general de anomalía
        self.anomaly_score = self._calculate_anomaly_score(anomalies)
        self.current_anomalies = anomalies
        
        # Actualizar historial
        self.pattern_history.append({
            'features': features,
            'anomalies': anomalies,
            'timestamp': current_time
        })
        
        return {
            'anomaly_score': self.anomaly_score,
            'anomaly_level': self._categorize_anomaly_level(self.anomaly_score),
            'detected_anomalies': self._get_detected_anomalies(anomalies),
            'confidence': self._calculate_detection_confidence(),
            'details': anomalies,
            'recommendations': self._generate_recommendations(anomalies)
        }
    
    def _extract_facial_features(self, landmarks):
        """Extrae características faciales para análisis"""
        features = {}
        
        # Ratios faciales
        features['eye_aspect_ratio'] = self._calculate_average_ear(landmarks)
        features['mouth_aspect_ratio'] = self._calculate_mar(landmarks)
        features['eyebrow_distance'] = self._calculate_eyebrow_distance(landmarks)
        
        # Simetría
        features['facial_symmetry'] = self._calculate_overall_symmetry(landmarks)
        
        # Posiciones relativas
        features['head_tilt'] = self._calculate_head_tilt(landmarks)
        features['eye_gaze'] = self._estimate_eye_gaze(landmarks)
        
        return features
    
    def _detect_facial_asymmetry(self, landmarks):
        """Detecta asimetrías faciales anómalas"""
        asymmetry_score = 0
        
        # Comparar lados izquierdo y derecho
        left_features = self._extract_left_side_features(landmarks)
        right_features = self._extract_right_side_features(landmarks)
        
        # Calcular diferencias
        for feature in ['eye_size', 'eyebrow_height', 'mouth_corner']:
            if feature in left_features and feature in right_features:
                diff = abs(left_features[feature] - right_features[feature])
                normalized_diff = diff / max(left_features[feature], right_features[feature], 0.01)
                asymmetry_score += normalized_diff
        
        # Normalizar score
        asymmetry_score = min(1.0, asymmetry_score / 3)
        
        return {
            'score': asymmetry_score,
            'detected': asymmetry_score > self.anomaly_types['asymmetry']['threshold'],
            'details': {
                'left_right_difference': asymmetry_score,
                'severity': 'high' if asymmetry_score > 0.5 else 'medium' if asymmetry_score > 0.3 else 'low'
            }
        }
    
    def _detect_unnatural_expression(self, features, emotion_data):
        """Detecta expresiones faciales no naturales o contradictorias"""
        unnaturalness_score = 0
        contradictions = []
        
        # Verificar coherencia entre características
        if features['mouth_aspect_ratio'] > 0.5 and features['eye_aspect_ratio'] < 0.2:
            unnaturalness_score += 0.3
            contradictions.append("Sonrisa con ojos cerrados")
        
        if features['eyebrow_distance'] < 0.8 and features['mouth_aspect_ratio'] > 0.3:
            unnaturalness_score += 0.2
            contradictions.append("Ceño fruncido con sonrisa")
        
        # Analizar coherencia con emociones detectadas
        if emotion_data:
            dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
            
            if dominant_emotion == 'alegria' and features['eyebrow_distance'] < 0.7:
                unnaturalness_score += 0.25
                contradictions.append("Alegría con tensión facial")
            
            if dominant_emotion in ['tristeza', 'enojo'] and features['mouth_aspect_ratio'] > 0.4:
                unnaturalness_score += 0.25
                contradictions.append(f"{dominant_emotion} con sonrisa")
        
        return {
            'score': min(1.0, unnaturalness_score),
            'detected': unnaturalness_score > self.anomaly_types['unnatural_expression']['threshold'],
            'details': {
                'contradictions': contradictions,
                'coherence_level': 1.0 - unnaturalness_score
            }
        }
    
    def _detect_irregular_eye_movement(self, landmarks):
        """Detecta movimientos oculares irregulares"""
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return {'score': 0, 'detected': False, 'details': {}}
        
        # Calcular centro de cada ojo
        left_center = self._get_eye_center(landmarks['left_eye'])
        right_center = self._get_eye_center(landmarks['right_eye'])
        
        # Agregar al historial
        self.eye_movement_tracker.append({
            'left': left_center,
            'right': right_center,
            'time': time.time()
        })
        
        irregularity_score = 0
        
        if len(self.eye_movement_tracker) >= 10:
            # Analizar patrones de movimiento
            movements = []
            for i in range(1, len(self.eye_movement_tracker)):
                prev = self.eye_movement_tracker[i-1]
                curr = self.eye_movement_tracker[i]
                
                # Calcular velocidad de movimiento
                left_movement = self._distance(prev['left'], curr['left'])
                right_movement = self._distance(prev['right'], curr['right'])
                time_diff = curr['time'] - prev['time']
                
                if time_diff > 0:
                    velocity = (left_movement + right_movement) / (2 * time_diff)
                    movements.append(velocity)
            
            if movements:
                # Detectar movimientos erráticos
                movement_std = np.std(movements)
                movement_mean = np.mean(movements)
                
                # Alta variabilidad = movimientos irregulares
                if movement_std > movement_mean * 0.5:
                    irregularity_score += 0.4
                
                # Movimientos muy rápidos
                if max(movements) > movement_mean * 3:
                    irregularity_score += 0.3
                
                # Asincronía entre ojos
                eye_sync = self._calculate_eye_synchronization()
                if eye_sync < 0.7:
                    irregularity_score += 0.3
        
        return {
            'score': min(1.0, irregularity_score),
            'detected': irregularity_score > self.anomaly_types['eye_movement']['threshold'],
            'details': {
                'movement_pattern': 'erratic' if irregularity_score > 0.5 else 'normal',
                'synchronization': self._calculate_eye_synchronization()
            }
        }
    
    def _detect_prolonged_smile(self, features, emotion_data, current_time):
        """Detecta sonrisas artificiales prolongadas"""
        smile_score = 0
        
        # Verificar si hay sonrisa
        is_smiling = features['mouth_aspect_ratio'] < 0.3 or \
                     (emotion_data and emotion_data.get('dominant_emotion') == 'alegria')
        
        if is_smiling:
            if self.smile_tracker['start_time'] is None:
                self.smile_tracker['start_time'] = current_time
            
            self.smile_tracker['duration'] = current_time - self.smile_tracker['start_time']
            
            # Sonrisa prolongada (más de 10 segundos)
            if self.smile_tracker['duration'] > 10:
                smile_score = min(1.0, (self.smile_tracker['duration'] - 10) / 20)
            
            # Verificar naturalidad de la sonrisa
            if features['eye_aspect_ratio'] > 0.28:  # Ojos no participan en la sonrisa
                smile_score += 0.3
        else:
            self.smile_tracker['start_time'] = None
            self.smile_tracker['duration'] = 0
        
        return {
            'score': min(1.0, smile_score),
            'detected': smile_score > self.anomaly_types['prolonged_smile']['threshold'],
            'details': {
                'duration': self.smile_tracker['duration'],
                'natural': smile_score < 0.3,
                'duchenne': features['eye_aspect_ratio'] < 0.25  # Sonrisa genuina incluye ojos
            }
        }
    
    def _detect_facial_rigidity(self, features):
        """Detecta rigidez facial (falta de micro-movimientos)"""
        rigidity_score = 0
        
        if len(self.pattern_history) >= 30:
            # Analizar variabilidad en las características
            recent_features = [p['features'] for p in list(self.pattern_history)[-30:]]
            
            # Calcular desviación estándar de cada característica
            feature_variations = {}
            for key in features.keys():
                values = [f.get(key, 0) for f in recent_features]
                feature_variations[key] = np.std(values)
            
            # Baja variabilidad indica rigidez
            avg_variation = np.mean(list(feature_variations.values()))
            
            if avg_variation < 0.02:
                rigidity_score = 0.8
            elif avg_variation < 0.05:
                rigidity_score = 0.5
            elif avg_variation < 0.1:
                rigidity_score = 0.3
        
        return {
            'score': rigidity_score,
            'detected': rigidity_score > self.anomaly_types['facial_rigidity']['threshold'],
            'details': {
                'movement_level': 1.0 - rigidity_score,
                'natural_movement': rigidity_score < 0.3
            }
        }
    
    def _detect_micro_expressions(self, emotion_data):
        """Detecta micro-expresiones (cambios emocionales muy rápidos)"""
        micro_expression_score = 0
        detected_micros = []
        
        if emotion_data:
            current_emotion = emotion_data.get('dominant_emotion', 'neutral')
            
            # Agregar al historial
            self.expression_change_tracker.append({
                'emotion': current_emotion,
                'time': time.time()
            })
            
            if len(self.expression_change_tracker) >= 10:
                # Analizar cambios rápidos
                changes = []
                for i in range(1, len(self.expression_change_tracker)):
                    if self.expression_change_tracker[i]['emotion'] != self.expression_change_tracker[i-1]['emotion']:
                        time_diff = self.expression_change_tracker[i]['time'] - self.expression_change_tracker[i-1]['time']
                        
                        # Micro-expresión: cambio en menos de 0.5 segundos
                        if time_diff < 0.5:
                            changes.append({
                                'from': self.expression_change_tracker[i-1]['emotion'],
                                'to': self.expression_change_tracker[i]['emotion'],
                                'duration': time_diff
                            })
                            micro_expression_score += 0.2
                
                detected_micros = changes[-3:]  # Últimas 3 micro-expresiones
        
        return {
            'score': min(1.0, micro_expression_score),
            'detected': micro_expression_score > self.anomaly_types['micro_expressions']['threshold'],
            'details': {
                'count': len(detected_micros),
                'recent_changes': detected_micros
            }
        }
    
    def _calculate_anomaly_score(self, anomalies):
        """Calcula score general de anomalía ponderado"""
        total_score = 0
        
        for anomaly_type, detection in anomalies.items():
            if anomaly_type in self.anomaly_types:
                weight = self.anomaly_types[anomaly_type]['weight']
                score = detection.get('score', 0)
                total_score += score * weight * self.sensitivity
        
        return min(100, int(total_score * 100))
    
    def _categorize_anomaly_level(self, score):
        """Categoriza el nivel de anomalía"""
        if score < 20:
            return 'normal'
        elif score < 40:
            return 'leve'
        elif score < 60:
            return 'moderado'
        elif score < 80:
            return 'alto'
        else:
            return 'crítico'
    
    def _get_detected_anomalies(self, anomalies):
        """Obtiene lista de anomalías detectadas"""
        detected = []
        
        for anomaly_type, detection in anomalies.items():
            if detection.get('detected', False):
                detected.append({
                    'type': anomaly_type,
                    'score': detection['score'],
                    'details': detection.get('details', {})
                })
        
        return detected
    
    def _calculate_detection_confidence(self):
        """Calcula confianza en la detección"""
        if len(self.pattern_history) < 30:
            return 0.5  # Confianza baja sin suficiente historial
        
        # Basada en consistencia de detecciones
        recent_scores = [p['anomalies'] for p in list(self.pattern_history)[-30:]]
        
        # Si las detecciones son consistentes, alta confianza
        consistency = 1.0 - np.std([self._calculate_anomaly_score(s) for s in recent_scores]) / 100
        
        return min(1.0, max(0.0, consistency))
    
    def _generate_recommendations(self, anomalies):
        """Genera recomendaciones basadas en anomalías detectadas"""
        recommendations = []
        
        if anomalies['asymmetry']['detected']:
            recommendations.append("Asimetría facial detectada - Verificar condición médica")
        
        if anomalies['unnatural_expression']['detected']:
            recommendations.append("Expresiones contradictorias - Evaluar estado emocional")
        
        if anomalies['eye_movement']['detected']:
            recommendations.append("Movimientos oculares irregulares - Posible fatiga o distracción")
        
        if anomalies['prolonged_smile']['detected']:
            recommendations.append("Sonrisa prolongada no natural - Verificar bienestar")
        
        if anomalies['facial_rigidity']['detected']:
            recommendations.append("Rigidez facial - Recomendar ejercicios de relajación")
        
        if anomalies['micro_expressions']['detected']:
            recommendations.append("Micro-expresiones detectadas - Posible estrés emocional")
        
        if not recommendations:
            recommendations.append("Comportamiento facial dentro de parámetros normales")
        
        return recommendations
    
    def draw_anomaly_info(self, frame, anomaly_data, position=(10, 400)):
        """Dibuja información de anomalías en el frame"""
        if not anomaly_data:
            return frame
        
        score = anomaly_data['anomaly_score']
        level = anomaly_data['anomaly_level']
        
        # Color según nivel
        if level == 'normal':
            color = self.anomaly_colors['low']
        elif level in ['leve', 'moderado']:
            color = self.anomaly_colors['medium']
        else:
            color = self.anomaly_colors['high']
        
        # Título
        title = f"Anomalias: {score}% ({level.upper()})"
        cv2.putText(frame, title, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Anomalías detectadas
        y_offset = position[1] + 30
        detected = anomaly_data['detected_anomalies']
        
        if detected:
            for i, anomaly in enumerate(detected[:3]):  # Máximo 3
                text = f"- {anomaly['type'].replace('_', ' ').title()}"
                cv2.putText(frame, text, (position[0] + 10, y_offset + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Comportamiento normal", 
                       (position[0] + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    # Métodos auxiliares
    def _calculate_average_ear(self, landmarks):
        """Calcula EAR promedio de ambos ojos"""
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
        
        top_lip = landmarks['top_lip']
        bottom_lip = landmarks['bottom_lip']
        
        # Distancia vertical
        A = self._distance(top_lip[3], bottom_lip[3])
        B = self._distance(top_lip[4], bottom_lip[4])
        
        # Distancia horizontal
        C = self._distance(top_lip[0], top_lip[6])
        
        return (A + B) / (2.0 * C) if C > 0 else 0
    
    def _calculate_eyebrow_distance(self, landmarks):
        """Calcula distancia normalizada entre cejas"""
        if 'left_eyebrow' not in landmarks or 'right_eyebrow' not in landmarks:
            return 1.0
        
        left_inner = landmarks['left_eyebrow'][-1]
        right_inner = landmarks['right_eyebrow'][0]
        
        distance = self._distance(left_inner, right_inner)
        
        # Normalizar por ancho de cara
        face_width = self._get_face_width(landmarks)
        
        return distance / face_width if face_width > 0 else 1.0
    
    def _calculate_overall_symmetry(self, landmarks):
        """Calcula simetría facial general"""
        # Implementación simplificada
        return 0.9  # Placeholder
    
    def _calculate_head_tilt(self, landmarks):
        """Calcula inclinación de la cabeza"""
        if 'nose_bridge' not in landmarks or len(landmarks['nose_bridge']) < 2:
            return 0
        
        # Usar línea de la nariz para estimar inclinación
        top = landmarks['nose_bridge'][0]
        bottom = landmarks['nose_bridge'][-1]
        
        angle = np.arctan2(bottom[1] - top[1], bottom[0] - top[0])
        
        # Convertir a grados y normalizar
        degrees = np.degrees(angle) - 90  # 0 grados = vertical
        
        return degrees
    
    def _estimate_eye_gaze(self, landmarks):
        """Estima dirección de la mirada"""
        # Implementación simplificada
        return {'horizontal': 0, 'vertical': 0}
    
    def _extract_left_side_features(self, landmarks):
        """Extrae características del lado izquierdo"""
        features = {}
        
        if 'left_eye' in landmarks:
            features['eye_size'] = self._calculate_eye_area(landmarks['left_eye'])
        
        if 'left_eyebrow' in landmarks:
            features['eyebrow_height'] = np.mean([p[1] for p in landmarks['left_eyebrow']])
        
        if 'top_lip' in landmarks:
            features['mouth_corner'] = landmarks['top_lip'][0][1]
        
        return features
    
    def _extract_right_side_features(self, landmarks):
        """Extrae características del lado derecho"""
        features = {}
        
        if 'right_eye' in landmarks:
            features['eye_size'] = self._calculate_eye_area(landmarks['right_eye'])
        
        if 'right_eyebrow' in landmarks:
            features['eyebrow_height'] = np.mean([p[1] for p in landmarks['right_eyebrow']])
        
        if 'top_lip' in landmarks:
            features['mouth_corner'] = landmarks['top_lip'][6][1]
        
        return features
    
    def _get_eye_center(self, eye_points):
        """Calcula centro del ojo"""
        if not eye_points:
            return (0, 0)
        
        x = int(np.mean([p[0] for p in eye_points]))
        y = int(np.mean([p[1] for p in eye_points]))
        
        return (x, y)
    
    def _calculate_eye_area(self, eye_points):
        """Calcula área del ojo"""
        if len(eye_points) < 6:
            return 0
        
        # Aproximación usando distancias
        height = (self._distance(eye_points[1], eye_points[5]) + 
                 self._distance(eye_points[2], eye_points[4])) / 2
        width = self._distance(eye_points[0], eye_points[3])
        
        return height * width
    
    def _calculate_eye_synchronization(self):
        """Calcula sincronización entre movimientos de ambos ojos"""
        if len(self.eye_movement_tracker) < 5:
            return 1.0
        
        # Calcular correlación entre movimientos
        left_movements = []
        right_movements = []
        
        for i in range(1, len(self.eye_movement_tracker)):
            prev = self.eye_movement_tracker[i-1]
            curr = self.eye_movement_tracker[i]
            
            left_movements.append(self._distance(prev['left'], curr['left']))
            right_movements.append(self._distance(prev['right'], curr['right']))
        
        if left_movements and right_movements:
            correlation = np.corrcoef(left_movements, right_movements)[0, 1]
            return abs(correlation)  # Valor entre 0 y 1
        
        return 1.0
    
    def _get_face_width(self, landmarks):
        """Obtiene ancho de la cara"""
        if 'chin' in landmarks and len(landmarks['chin']) >= 17:
            return self._distance(landmarks['chin'][0], landmarks['chin'][16])
        return 100
    
    def _distance(self, p1, p2):
        """Calcula distancia entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _update_baseline(self, features):
        """Actualiza el baseline de comportamiento normal"""
        if len(self.pattern_history) >= 60:  # 2 segundos de datos
            self.baseline_patterns = features.copy()
            self.baseline_established = True
    
    def _get_default_result(self):
        """Resultado por defecto"""
        return {
            'anomaly_score': 0,
            'anomaly_level': 'normal',
            'detected_anomalies': [],
            'confidence': 0,
            'details': {},
            'recommendations': ["Insuficientes datos para análisis"]
        }
    
    def get_anomaly_report(self):
        """Genera reporte de anomalías"""
        return {
            'current_score': self.anomaly_score,
            'level': self._categorize_anomaly_level(self.anomaly_score),
            'active_anomalies': self._get_detected_anomalies(self.current_anomalies),
            'baseline_established': self.baseline_established,
            'history_length': len(self.pattern_history),
            'recommendations': self._generate_recommendations(self.current_anomalies)
        }
    
    def reset(self):
        """Reinicia el detector"""
        self.pattern_history.clear()
        self.baseline_established = False
        self.baseline_patterns = {}
        self.smile_tracker = {'start_time': None, 'duration': 0}
        self.eye_movement_tracker.clear()
        self.expression_change_tracker.clear()
        self.anomaly_score = 0