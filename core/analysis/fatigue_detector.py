"""
Módulo de Detección de Fatiga
=============================
Detecta niveles de fatiga basándose en apertura de ojos y microsueños.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging

class FatigueDetector:
    def __init__(self, headless=False):
        """
        Inicializa el detector de fatiga.
        
        Args:
            headless: Si True, desactiva visualizaciones (modo servidor)
        """
        self.logger = logging.getLogger('FatigueDetector')
        self.headless = headless
        
        # Estado actual
        self.fatigue_level = 0
        self.baseline = None
        self.is_calibrated = False
        
        # Historial para suavizado
        self.eye_openness_history = deque(maxlen=30)  # ~1 segundo a 30fps
        self.fatigue_history = deque(maxlen=10)
        
        # Detección de microsueños
        self.eyes_closed_start = None
        self.microsleep_threshold = 0.5  # segundos
        self.microsleep_count = 0
        self.last_microsleep_time = 0
        
        # Configuración visual (solo si no es headless)
        if not self.headless:
            self.bar_color = (251, 146, 60)  # Naranja
            self.bar_width = 200
            self.bar_height = 30
        
        # Timestamp
        self.last_update = time.time()
        
    def set_baseline(self, baseline):
        """
        Configura el baseline del operador actual.
        
        Args:
            baseline: Diccionario con métricas calibradas del operador
        """
        if baseline and 'facial_metrics' in baseline:
            self.baseline = baseline
            
            # Extraer umbrales personalizados
            eye_metrics = baseline['facial_metrics']['eye_measurements']
            self.normal_eye_openness = eye_metrics['eye_openness_avg']
            
            # Calcular umbrales adaptativos
            thresholds = baseline.get('thresholds', {}).get('fatigue', {})
            self.microsleep_eye_threshold = thresholds.get('microsleep_threshold', self.normal_eye_openness * 0.6)
            self.severe_fatigue_threshold = thresholds.get('severe_fatigue_threshold', self.normal_eye_openness * 0.4)
            
            self.is_calibrated = True
            self.logger.info(f"Baseline configurado - Apertura normal: {self.normal_eye_openness:.3f}")
        else:
            self.logger.warning("Baseline inválido, usando valores por defecto")
            self._set_default_values()
    
    def _set_default_values(self):
        """Establece valores por defecto cuando no hay calibración"""
        self.normal_eye_openness = 0.25
        self.microsleep_eye_threshold = 0.15
        self.severe_fatigue_threshold = 0.10
        self.is_calibrated = False
    
    def analyze(self, face_landmarks):
        """
        Analiza el nivel de fatiga basado en landmarks faciales.
        
        Args:
            face_landmarks: Diccionario con landmarks faciales
            
        Returns:
            dict: Resultados del análisis
        """
        self.last_update = time.time()
        
        if not face_landmarks:
            return self._get_default_result()
        
        # Si no hay baseline, usar valores por defecto
        if not self.is_calibrated:
            self._set_default_values()
        
        # Calcular apertura de ojos
        current_eye_openness = self._calculate_eye_openness(face_landmarks)
        
        if current_eye_openness >= 0:
            self.eye_openness_history.append(current_eye_openness)
            
            # Detectar microsueños
            self._detect_microsleep(current_eye_openness)
            
            # Calcular nivel de fatiga
            fatigue_score = self._calculate_fatigue_score(current_eye_openness)
            
            # Suavizar con historial
            self.fatigue_history.append(fatigue_score)
            self.fatigue_level = int(np.mean(self.fatigue_history))
        
        return {
            'fatigue_percentage': self.fatigue_level,
            'is_fatigued': self.fatigue_level > 60,
            'is_critical': self.fatigue_level > 80,
            'microsleep_detected': self.microsleep_count > 0,
            'microsleep_count': self.microsleep_count,
            'eye_openness': current_eye_openness if current_eye_openness >= 0 else None,
            'is_calibrated': self.is_calibrated
        }
    
    def _calculate_eye_openness(self, landmarks):
        """Calcula la apertura promedio de ambos ojos"""
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return -1
        
        try:
            left_ear = self._calculate_ear(landmarks['left_eye'])
            right_ear = self._calculate_ear(landmarks['right_eye'])
            
            # Promedio de ambos ojos
            avg_ear = (left_ear + right_ear) / 2
            return avg_ear
            
        except Exception as e:
            self.logger.error(f"Error calculando apertura de ojos: {e}")
            return -1
    
    def _calculate_ear(self, eye_points):
        """
        Calcula Eye Aspect Ratio (EAR) para un ojo.
        Mayor EAR = ojo más abierto
        """
        if len(eye_points) != 6:
            return 0.25  # Valor por defecto
        
        try:
            # Distancias verticales
            v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            
            # Distancia horizontal
            h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            
            if h == 0:
                return 0
                
            ear = (v1 + v2) / (2.0 * h)
            return ear
            
        except Exception:
            return 0.25
    
    def _detect_microsleep(self, current_eye_openness):
        """Detecta microsueños (ojos cerrados por más de 0.5 segundos)"""
        current_time = time.time()
        
        # Verificar si los ojos están cerrados
        if current_eye_openness < self.microsleep_eye_threshold:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            else:
                # Calcular duración con ojos cerrados
                closed_duration = current_time - self.eyes_closed_start
                
                if closed_duration >= self.microsleep_threshold:
                    # Microsueño detectado
                    if current_time - self.last_microsleep_time > 2:  # Evitar conteo múltiple
                        self.microsleep_count += 1
                        self.last_microsleep_time = current_time
                        self.logger.warning(f"Microsueño detectado #{self.microsleep_count}")
        else:
            # Ojos abiertos, resetear contador
            self.eyes_closed_start = None
        
        # Resetear contador después de 5 minutos sin microsueños
        if current_time - self.last_microsleep_time > 300:
            self.microsleep_count = 0
    
    def _calculate_fatigue_score(self, current_eye_openness):
        """
        Calcula el nivel de fatiga (0-100).
        Basado en la desviación del baseline personal.
        """
        if current_eye_openness < 0:
            return self.fatigue_level  # Mantener valor anterior
        
        # Factor 1: Desviación del baseline
        if self.is_calibrated:
            # Calcular qué tan cerrados están los ojos respecto a lo normal
            deviation = (self.normal_eye_openness - current_eye_openness) / self.normal_eye_openness
            deviation = max(0, deviation)  # Solo valores positivos
            
            # Convertir a escala 0-100
            base_fatigue = min(100, deviation * 150)
        else:
            # Sin calibración, usar escala fija
            if current_eye_openness < 0.10:
                base_fatigue = 90
            elif current_eye_openness < 0.15:
                base_fatigue = 70
            elif current_eye_openness < 0.20:
                base_fatigue = 50
            elif current_eye_openness < 0.25:
                base_fatigue = 30
            else:
                base_fatigue = 10
        
        # Factor 2: Microsueños (aumenta fatiga significativamente)
        microsleep_penalty = min(40, self.microsleep_count * 15)
        
        # Factor 3: Variabilidad (ojos que se abren y cierran mucho = fatiga)
        if len(self.eye_openness_history) > 10:
            variability = np.std(list(self.eye_openness_history)[-10:])
            variability_penalty = min(20, variability * 100)
        else:
            variability_penalty = 0
        
        # Combinar factores
        total_fatigue = base_fatigue + microsleep_penalty + variability_penalty
        
        # Limitar a 0-100
        return min(100, max(0, total_fatigue))
    
    def draw_fatigue_bar(self, frame, position=(None, None)):
        """
        Dibuja solo una barra de fatiga con porcentaje.
        
        Args:
            frame: Frame donde dibujar
            position: (x, y) posición de la barra
            
        Returns:
            frame: Frame con barra dibujada
        """
        if self.headless:
            return frame
        
        h, w = frame.shape[:2]
        
        # Posición por defecto: arriba a la derecha
        if position[0] is None:
            bar_x = w - self.bar_width - 20
        else:
            bar_x = position[0]
            
        if position[1] is None:
            bar_y = 20
        else:
            bar_y = position[1]
        
        # Texto de fatiga
        text = f"FATIGA: {self.fatigue_level}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Color basado en nivel
        if self.fatigue_level < 40:
            color = (0, 255, 0)  # Verde
        elif self.fatigue_level < 70:
            color = (0, 165, 255)  # Naranja
        else:
            color = (0, 0, 255)  # Rojo
        
        # Dibujar texto
        cv2.putText(frame, text, (bar_x, bar_y - 5),
                   font, font_scale, color, thickness)
        
        # Fondo de la barra
        cv2.rectangle(frame, 
                     (bar_x, bar_y),
                     (bar_x + self.bar_width, bar_y + self.bar_height),
                     (50, 50, 50), -1)
        
        # Barra de progreso
        progress_width = int(self.bar_width * (self.fatigue_level / 100))
        if progress_width > 0:
            cv2.rectangle(frame,
                         (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + self.bar_height),
                         color, -1)
        
        # Borde
        cv2.rectangle(frame,
                     (bar_x, bar_y),
                     (bar_x + self.bar_width, bar_y + self.bar_height),
                     (200, 200, 200), 1)
        
        # Indicador de microsueño si está activo
        if self.microsleep_count > 0:
            alert_text = f"⚠️ Microsueños: {self.microsleep_count}"
            cv2.putText(frame, alert_text, (bar_x, bar_y + self.bar_height + 20),
                       font, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def get_fatigue_report_for_server(self):
        """
        Genera reporte JSON para el servidor.
        
        Returns:
            dict: Reporte de fatiga en formato JSON
        """
        recent_openness = list(self.eye_openness_history)[-10:] if self.eye_openness_history else []
        
        return {
            'timestamp': self.last_update,
            'fatigue_level': self.fatigue_level,
            'status': self._get_fatigue_status(),
            'metrics': {
                'current_eye_openness': recent_openness[-1] if recent_openness else None,
                'average_eye_openness': np.mean(recent_openness) if recent_openness else None,
                'microsleep_count': self.microsleep_count,
                'is_calibrated': self.is_calibrated
            },
            'thresholds': {
                'normal_openness': self.normal_eye_openness,
                'microsleep_threshold': self.microsleep_eye_threshold
            },
            'alerts': self._generate_alerts()
        }
    
    def _get_fatigue_status(self):
        """Determina el estado de fatiga"""
        if self.fatigue_level < 30:
            return 'normal'
        elif self.fatigue_level < 60:
            return 'mild'
        elif self.fatigue_level < 80:
            return 'moderate'
        else:
            return 'severe'
    
    def _generate_alerts(self):
        """Genera alertas basadas en el nivel de fatiga"""
        alerts = []
        
        if self.fatigue_level >= 80:
            alerts.append({
                'type': 'critical_fatigue',
                'message': f'Fatiga crítica detectada: {self.fatigue_level}%',
                'level': 'critical'
            })
        elif self.fatigue_level >= 60:
            alerts.append({
                'type': 'high_fatigue',
                'message': f'Fatiga elevada: {self.fatigue_level}%',
                'level': 'warning'
            })
        
        if self.microsleep_count > 0:
            alerts.append({
                'type': 'microsleep',
                'message': f'{self.microsleep_count} microsueños detectados',
                'level': 'critical' if self.microsleep_count > 2 else 'warning'
            })
        
        return alerts
    
    def _get_default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'fatigue_percentage': self.fatigue_level,
            'is_fatigued': False,
            'is_critical': False,
            'microsleep_detected': False,
            'microsleep_count': 0,
            'eye_openness': None,
            'is_calibrated': self.is_calibrated
        }
    
    def reset(self):
        """Reinicia el detector"""
        self.fatigue_level = 0
        self.eye_openness_history.clear()
        self.fatigue_history.clear()
        self.microsleep_count = 0
        self.eyes_closed_start = None
        self.last_microsleep_time = 0