"""
Módulo de Análisis de Estrés
============================
Detecta niveles de estrés basado en micro-movimientos y tensión facial.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging

class StressAnalyzer:
    def __init__(self, time_window=30):
        """
        Inicializa el analizador de estrés.
        
        Args:
            time_window: Ventana de tiempo en segundos para análisis
        """
        self.logger = logging.getLogger('StressAnalyzer')
        
        # Configuración
        self.time_window = time_window
        self.facial_tension_threshold = 0.6
        self.micro_movement_threshold = 0.3
        
        # Historial de datos
        self.movement_history = deque(maxlen=30)
        self.tension_history = deque(maxlen=30)
        self.timestamps = deque(maxlen=30)
        
        # Estado
        self.last_landmarks = None
        self.stress_level = 0
        self.stress_indicators = {
            'tension': 0,
            'movement': 0,
            'stability': 100,
            'eye_strain': 0
        }
        
        # Colores para visualización
        self.stress_colors = {
            'low': (0, 255, 0),      # Verde
            'medium': (0, 165, 255),  # Naranja
            'high': (0, 0, 255)       # Rojo
        }
        
    def analyze(self, frame, face_landmarks):
        """
        Analiza el nivel de estrés del operador.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales detectados
            
        Returns:
            dict: Resultados del análisis de estrés
        """
        current_time = time.time()
        
        # Calcular indicadores
        tension = self._calculate_facial_tension(face_landmarks)
        movement = self._calculate_micro_movements(face_landmarks)
        eye_strain = self._calculate_eye_strain(face_landmarks)
        stability = self._calculate_stability()
        
        # Actualizar historial
        self.tension_history.append(tension)
        self.movement_history.append(movement)
        self.timestamps.append(current_time)
        
        # Actualizar landmarks previos
        self.last_landmarks = self._copy_landmarks(face_landmarks)
        
        # Limpiar historial antiguo
        self._clean_old_data(current_time)
        
        # Calcular nivel de estrés general
        self.stress_level = self._calculate_overall_stress(
            tension, movement, eye_strain, stability
        )
        
        # Actualizar indicadores
        self.stress_indicators = {
            'tension': int(tension * 100),
            'movement': int(movement * 100),
            'stability': int(stability * 100),
            'eye_strain': int(eye_strain * 100)
        }
        
        return {
            'stress_level': self.stress_level,
            'stress_category': self._categorize_stress(self.stress_level),
            'indicators': self.stress_indicators,
            'color': self._get_stress_color(self.stress_level),
            'recommendations': self._get_recommendations(self.stress_level)
        }
    
    def _calculate_facial_tension(self, landmarks):
        """Calcula tensión facial basada en músculos faciales"""
        if not landmarks:
            return 0.0
        
        tension_indicators = []
        
        # 1. Tensión en la mandíbula
        jaw_width = self._distance(landmarks['chin'][0], landmarks['chin'][16])
        jaw_center = landmarks['chin'][8]
        jaw_tension = self._calculate_jaw_tension(landmarks['chin'], jaw_center)
        tension_indicators.append(jaw_tension)
        
        # 2. Tensión en las cejas (fruncir el ceño)
        left_brow = landmarks.get('left_eyebrow', [])
        right_brow = landmarks.get('right_eyebrow', [])
        if left_brow and right_brow:
            brow_distance = self._distance(left_brow[-1], right_brow[0])
            brow_tension = 1.0 - (brow_distance / jaw_width) if jaw_width > 0 else 0
            tension_indicators.append(min(1.0, max(0.0, brow_tension)))
        
        # 3. Tensión en los labios
        if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
            lip_compression = self._calculate_lip_compression(
                landmarks['top_lip'], 
                landmarks['bottom_lip']
            )
            tension_indicators.append(lip_compression)
        
        # 4. Tensión alrededor de los ojos
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            eye_tension = self._calculate_eye_tension(
                landmarks['left_eye'],
                landmarks['right_eye']
            )
            tension_indicators.append(eye_tension)
        
        # Promedio ponderado de indicadores
        if tension_indicators:
            return sum(tension_indicators) / len(tension_indicators)
        return 0.0
    
    def _calculate_micro_movements(self, landmarks):
        """Detecta micro-movimientos nerviosos"""
        if not self.last_landmarks or not landmarks:
            return 0.0
        
        movements = []
        
        # Comparar puntos clave entre frames
        for feature in ['nose_tip', 'chin', 'left_eye', 'right_eye']:
            if feature in landmarks and feature in self.last_landmarks:
                current_points = landmarks[feature]
                last_points = self.last_landmarks[feature]
                
                if len(current_points) == len(last_points):
                    # Calcular movimiento promedio
                    feature_movement = self._calculate_feature_movement(
                        current_points, last_points
                    )
                    movements.append(feature_movement)
        
        if movements:
            # Normalizar por el tamaño de la cara
            face_width = self._get_face_width(landmarks)
            avg_movement = sum(movements) / len(movements)
            normalized_movement = avg_movement / face_width if face_width > 0 else 0
            
            # Amplificar pequeños movimientos
            return min(1.0, normalized_movement * 50)
        
        return 0.0
    
    def _calculate_eye_strain(self, landmarks):
        """Calcula tensión ocular"""
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return 0.0
        
        strain_indicators = []
        
        # 1. Parpadeo frecuente (usando historial si está disponible)
        left_ear = self._calculate_ear(landmarks['left_eye'])
        right_ear = self._calculate_ear(landmarks['right_eye'])
        avg_ear = (left_ear + right_ear) / 2
        
        # Ojos más cerrados = más tensión
        eye_closure = 1.0 - min(1.0, avg_ear / 0.3)
        strain_indicators.append(eye_closure)
        
        # 2. Asimetría ocular (un ojo más cerrado que otro)
        asymmetry = abs(left_ear - right_ear) / max(left_ear, right_ear, 0.01)
        strain_indicators.append(min(1.0, asymmetry * 2))
        
        return sum(strain_indicators) / len(strain_indicators)
    
    def _calculate_stability(self):
        """Calcula estabilidad general (inverso de variabilidad)"""
        if len(self.movement_history) < 5:
            return 1.0  # Asume estable si no hay suficiente historial
        
        # Calcular variabilidad en el movimiento
        recent_movements = list(self.movement_history)[-10:]
        if recent_movements:
            std_dev = np.std(recent_movements)
            # Invertir y normalizar (más variabilidad = menos estabilidad)
            stability = 1.0 - min(1.0, std_dev * 5)
            return stability
        
        return 1.0
    
    def _calculate_overall_stress(self, tension, movement, eye_strain, stability):
        """Calcula nivel de estrés general ponderado"""
        # Pesos para cada componente
        weights = {
            'tension': 0.35,
            'movement': 0.25,
            'eye_strain': 0.20,
            'instability': 0.20  # Inverso de estabilidad
        }
        
        # Calcular score ponderado
        stress_score = (
            tension * weights['tension'] +
            movement * weights['movement'] +
            eye_strain * weights['eye_strain'] +
            (1 - stability) * weights['instability']
        )
        
        # Aplicar factor de sensibilidad y limitar a 0-100
        stress_level = int(min(100, max(0, stress_score * 100)))
        
        return stress_level
    
    def _categorize_stress(self, level):
        """Categoriza el nivel de estrés"""
        if level < 30:
            return "bajo"
        elif level < 60:
            return "moderado"
        elif level < 80:
            return "alto"
        else:
            return "crítico"
    
    def _get_stress_color(self, level):
        """Obtiene color según nivel de estrés"""
        if level < 30:
            return self.stress_colors['low']
        elif level < 60:
            return self.stress_colors['medium']
        else:
            return self.stress_colors['high']
    
    def _get_recommendations(self, level):
        """Genera recomendaciones según el nivel de estrés"""
        if level < 30:
            return ["Nivel de estrés bajo", "Continúe con su trabajo normalmente"]
        elif level < 60:
            return ["Tome respiraciones profundas", "Realice pausas cortas cada hora"]
        elif level < 80:
            return ["Considere tomar un descanso de 10-15 minutos", 
                   "Realice ejercicios de relajación"]
        else:
            return ["ALTO ESTRÉS DETECTADO", 
                   "Tome un descanso inmediato",
                   "Consulte con su supervisor"]
    
    def draw_stress_info(self, frame, stress_data, position=(10, 200)):
        """Dibuja información de estrés en el frame"""
        if not stress_data:
            return frame
        
        level = stress_data['stress_level']
        category = stress_data['stress_category']
        color = stress_data['color']
        
        # Título
        title = f"Estres: {level}% ({category.upper()})"
        cv2.putText(frame, title, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barra de progreso
        self._draw_progress_bar(frame, level, position, color)
        
        # Indicadores detallados
        y_offset = position[1] + 60
        indicators = stress_data['indicators']
        
        for i, (name, value) in enumerate(indicators.items()):
            text = f"{name.title()}: {value}%"
            cv2.putText(frame, text, (position[0], y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_progress_bar(self, frame, level, position, color):
        """Dibuja barra de progreso para el nivel de estrés"""
        bar_width = 200
        bar_height = 15
        x, y = position[0], position[1] + 25
        
        # Fondo
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     (100, 100, 100), -1)
        
        # Barra de progreso
        progress_width = int(bar_width * (level / 100))
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height),
                     color, -1)
        
        # Marcadores de nivel
        for threshold, level_name in [(30, "Bajo"), (60, "Medio"), (80, "Alto")]:
            marker_x = x + int(bar_width * (threshold / 100))
            cv2.line(frame, (marker_x, y-5), (marker_x, y+bar_height+5),
                    (200, 200, 200), 1)
    
    # Métodos auxiliares
    def _distance(self, p1, p2):
        """Calcula distancia entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_jaw_tension(self, jaw_points, center):
        """Calcula tensión en la mandíbula"""
        if len(jaw_points) < 17:
            return 0.0
        
        # Calcular desviación de la forma ideal relajada
        deviations = []
        for i in range(len(jaw_points)):
            expected_angle = i * (180 / (len(jaw_points) - 1))
            actual_angle = self._calculate_angle(center, jaw_points[i])
            deviation = abs(expected_angle - actual_angle)
            deviations.append(deviation)
        
        avg_deviation = sum(deviations) / len(deviations) if deviations else 0
        return min(1.0, avg_deviation / 45)  # Normalizar a 0-1
    
    def _calculate_lip_compression(self, top_lip, bottom_lip):
        """Calcula compresión de labios"""
        if not top_lip or not bottom_lip:
            return 0.0
        
        # Distancia promedio entre labios
        distances = []
        for i in range(min(len(top_lip), len(bottom_lip))):
            dist = self._distance(top_lip[i], bottom_lip[i])
            distances.append(dist)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        mouth_width = self._distance(top_lip[0], top_lip[-1])
        
        # Normalizar por ancho de boca
        if mouth_width > 0:
            compression = 1.0 - min(1.0, (avg_distance / mouth_width) * 5)
            return compression
        
        return 0.0
    
    def _calculate_eye_tension(self, left_eye, right_eye):
        """Calcula tensión alrededor de los ojos"""
        tensions = []
        
        for eye_points in [left_eye, right_eye]:
            if len(eye_points) >= 6:
                # Calcular "squint" factor
                vertical_dist = (self._distance(eye_points[1], eye_points[5]) +
                               self._distance(eye_points[2], eye_points[4])) / 2
                horizontal_dist = self._distance(eye_points[0], eye_points[3])
                
                if horizontal_dist > 0:
                    squint = 1.0 - min(1.0, (vertical_dist / horizontal_dist) * 3)
                    tensions.append(squint)
        
        return sum(tensions) / len(tensions) if tensions else 0.0
    
    def _calculate_feature_movement(self, current_points, last_points):
        """Calcula movimiento entre puntos de una característica"""
        total_movement = 0
        count = 0
        
        for curr, last in zip(current_points, last_points):
            movement = self._distance(curr, last)
            total_movement += movement
            count += 1
        
        return total_movement / count if count > 0 else 0
    
    def _get_face_width(self, landmarks):
        """Obtiene ancho de la cara para normalización"""
        if 'chin' in landmarks and len(landmarks['chin']) >= 17:
            return self._distance(landmarks['chin'][0], landmarks['chin'][16])
        return 100  # Valor por defecto
    
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0.3
        
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        h = self._distance(eye_points[0], eye_points[3])
        
        return (v1 + v2) / (2.0 * h) if h > 0 else 0
    
    def _calculate_angle(self, center, point):
        """Calcula ángulo entre centro y punto"""
        return np.degrees(np.arctan2(point[1] - center[1], point[0] - center[0]))
    
    def _copy_landmarks(self, landmarks):
        """Crea una copia profunda de los landmarks"""
        if not landmarks:
            return None
        
        copy = {}
        for feature, points in landmarks.items():
            copy[feature] = [tuple(p) for p in points]
        return copy
    
    def _clean_old_data(self, current_time):
        """Limpia datos fuera de la ventana de tiempo"""
        cutoff_time = current_time - self.time_window
        
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            if self.movement_history:
                self.movement_history.popleft()
            if self.tension_history:
                self.tension_history.popleft()
    
    def get_stress_report(self):
        """Genera reporte detallado del estado de estrés"""
        return {
            'current_level': self.stress_level,
            'category': self._categorize_stress(self.stress_level),
            'indicators': self.stress_indicators,
            'history_length': len(self.timestamps),
            'avg_movement': np.mean(self.movement_history) if self.movement_history else 0,
            'avg_tension': np.mean(self.tension_history) if self.tension_history else 0,
            'recommendations': self._get_recommendations(self.stress_level)
        }
    
    def reset(self):
        """Reinicia el analizador"""
        self.movement_history.clear()
        self.tension_history.clear()
        self.timestamps.clear()
        self.last_landmarks = None
        self.stress_level = 0