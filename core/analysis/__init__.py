"""
Módulo de Análisis Avanzado
===========================
Proporciona análisis facial avanzado para el sistema de seguridad.
"""

import cv2
import numpy as np
from collections import deque
import logging
import time

# Importar los analizadores específicos
from .emotion_analyzer import EmotionAnalyzer
from .stress_analyzer import StressAnalyzer
from .fatigue_stress_monitor import FatigueStressMonitor
from .pulse_estimator import PulseEstimator
from .anomaly_detector import AnomalyDetector
from .analysis_dashboard import AnalysisDashboard


# Versión del módulo
__version__ = "1.0.0"

# Exportar clases principales
__all__ = [
    'EmotionAnalyzer',
    'StressAnalyzer', 
    'FatigueStressMonitor',
    'PulseEstimator',
    'AnomalyDetector',
    'AnalysisDashboard',    
    'BaseAnalyzer',
    'IntegratedAnalysisSystem'
]


class BaseAnalyzer:
    """
    Clase base para todos los analizadores.
    Proporciona funcionalidad común.
    """
    
    def __init__(self):
        """Inicializa el analizador base"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Estado de iluminación
        self.is_night_mode = False
        self.light_level = 0
        self.night_mode_threshold = 50
        
        # Historial genérico
        self.history = deque(maxlen=300)  # 10 segundos a 30fps
        
        # Configuración de adaptación
        self.enable_auto_adaptation = True
        self.adaptation_factor = 1.0
        
    def detect_lighting_conditions(self, frame):
        """
        Auto-detección de condiciones de luz.
        
        Args:
            frame: Imagen para analizar
            
        Returns:
            dict: Información sobre condiciones de luz
        """
        # Convertir a escala de grises si es necesario
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calcular nivel promedio de iluminación
        self.light_level = np.mean(gray)
        
        # Determinar modo día/noche
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.night_mode_threshold
        
        # Log cambio de modo
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
            self.logger.info(f"Cambio a modo {mode_str} (Nivel: {self.light_level:.1f})")
        
        return {
            'light_level': self.light_level,
            'is_night_mode': self.is_night_mode,
            'mode': 'night' if self.is_night_mode else 'day',
            'quality': self._assess_light_quality()
        }
    
    def adjust_thresholds(self, base_threshold, adjustment_factor=0.1):
        """
        Ajuste dinámico de umbrales según iluminación.
        
        Args:
            base_threshold: Umbral base
            adjustment_factor: Factor de ajuste (0-1)
            
        Returns:
            float: Umbral ajustado
        """
        if not self.enable_auto_adaptation:
            return base_threshold
        
        # En modo nocturno, ser más permisivo
        if self.is_night_mode:
            # Reducir umbral proporcionalmente a la oscuridad
            darkness_factor = 1.0 - (self.light_level / self.night_mode_threshold)
            adjustment = base_threshold * adjustment_factor * darkness_factor
            return base_threshold - adjustment
        
        # En luz muy brillante, ser más estricto
        elif self.light_level > 200:
            brightness_factor = (self.light_level - 200) / 55  # Normalizar exceso
            adjustment = base_threshold * adjustment_factor * brightness_factor * 0.5
            return base_threshold + adjustment
        
        # Condiciones normales
        return base_threshold
    
    def _assess_light_quality(self):
        """Evalúa la calidad de iluminación"""
        if self.light_level < 30:
            return 'very_poor'
        elif self.light_level < 50:
            return 'poor'
        elif self.light_level < 150:
            return 'good'
        elif self.light_level < 200:
            return 'excellent'
        else:
            return 'overexposed'
    
    def preprocess_frame(self, frame):
        """
        Pre-procesa el frame según condiciones de luz.
        
        Args:
            frame: Imagen original
            
        Returns:
            frame: Imagen mejorada
        """
        # Detectar condiciones actuales
        self.detect_lighting_conditions(frame)
        
        # Aplicar mejoras según modo
        if self.is_night_mode:
            # Mejoras para modo nocturno/IR
            enhanced = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Reducir ruido
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
        
        elif self.light_level > 200:
            # Reducir sobreexposición
            return cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0)
        
        # Condiciones normales - mejora suave
        return frame
    
    def get_mode_info(self):
        """Obtiene información del modo actual"""
        return {
            'mode': 'NOCTURNO' if self.is_night_mode else 'DIURNO',
            'light_level': self.light_level,
            'adaptation_active': self.enable_auto_adaptation,
            'color': (0, 150, 255) if self.is_night_mode else (255, 200, 0)
        }


class IntegratedAnalysisSystem:
    """
    Sistema integrado que combina todos los análisis.
    """
    
    def __init__(self):
        """Inicializa el sistema integrado"""
        self.logger = logging.getLogger('IntegratedAnalysisSystem')
        
        # Inicializar analizadores
        self.emotion_analyzer = EmotionAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        self.fatigue_monitor = FatigueStressMonitor()
        self.anomaly_detector = AnomalyDetector()  # Agregar detector de anomalías
        self.pulse_estimator = PulseEstimator()    # Agregar estimador de pulso
            
        # Estado general
        self.operator_profile = {
            'emotional_state': 'neutral',
            'stress_level': 0,
            'fatigue_level': 0,
            'overall_condition': 'good'
        }
        
        # Historial para tendencias
        self.analysis_history = deque(maxlen=100)
        
    def analyze_operator(self, frame, face_landmarks, face_location=None):
        """
        Realiza análisis completo del operador.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales
            face_location: Ubicación del rostro
            
        Returns:
            dict: Análisis completo integrado
        """
        # Análisis individual
        emotion_result = self.emotion_analyzer.analyze(frame, face_landmarks)
        stress_result = self.stress_analyzer.analyze(frame, face_landmarks)
        
        # Para fatiga, usar el monitor simple
        fatigue_info = {'microsleep_count': 0}  # Simulado por ahora
        fatigue_monitor_result = self.fatigue_monitor.update(frame, face_landmarks, fatigue_info)
        
        # Convertir resultado del monitor a formato esperado
        fatigue_result = {
            'fatigue_score': fatigue_monitor_result['fatigue_level'],
            'fatigue_level': fatigue_monitor_result['fatigue_category'],
            'recommendations': self.fatigue_monitor.get_recommendations()
        }
        
        # Análisis de anomalías
        anomaly_result = self.anomaly_detector.analyze(frame, face_landmarks, emotion_result)
        
        # Análisis de pulso
        pulse_result = self.pulse_estimator.process_frame(frame, face_landmarks)
        
        # Combinar resultados
        integrated_result = {
            'emotion': emotion_result,
            'stress': stress_result,
            'fatigue': fatigue_result,
            'anomaly': anomaly_result,
            'pulse': pulse_result,
            'overall_assessment': self._calculate_overall_assessment(
                emotion_result, stress_result, fatigue_result
            ),
            'alerts': self._generate_alerts(emotion_result, stress_result, fatigue_result),
            'recommendations': self._generate_recommendations(
                emotion_result, stress_result, fatigue_result
            )
        }
        
        # Actualizar perfil del operador
        self._update_operator_profile(integrated_result)
        
        # Agregar al historial
        self.analysis_history.append(integrated_result)
        
        return integrated_result
    
    def _calculate_overall_assessment(self, emotion, stress, fatigue):
        """Calcula evaluación general del operador"""
        # Puntajes ponderados
        emotion_score = emotion['wellbeing'] * 0.3
        stress_score = (100 - stress['stress_level']) * 0.35
        fatigue_score = (100 - fatigue['fatigue_score']) * 0.35
        
        overall_score = emotion_score + stress_score + fatigue_score
        
        if overall_score > 80:
            return {
                'status': 'EXCELENTE',
                'score': overall_score,
                'color': (0, 255, 0)
            }
        elif overall_score > 60:
            return {
                'status': 'BUENO',
                'score': overall_score,
                'color': (0, 255, 255)
            }
        elif overall_score > 40:
            return {
                'status': 'REGULAR',
                'score': overall_score,
                'color': (0, 165, 255)
            }
        else:
            return {
                'status': 'REQUIERE ATENCIÓN',
                'score': overall_score,
                'color': (0, 0, 255)
            }
    
    def _generate_alerts(self, emotion, stress, fatigue):
        """Genera alertas basadas en los análisis"""
        alerts = []
        
        # Alertas emocionales
        if emotion['stress_level'] > 70:
            alerts.append({
                'type': 'emotion',
                'severity': 'high',
                'message': 'Alto estrés emocional detectado'
            })
        
        if emotion['anomalies'] > 50:
            alerts.append({
                'type': 'anomaly',
                'severity': 'medium',
                'message': 'Comportamiento facial anómalo'
            })
        
        # Alertas de estrés
        if stress['stress_level'] > 80:
            alerts.append({
                'type': 'stress',
                'severity': 'critical',
                'message': 'Nivel crítico de estrés'
            })
        
        # Alertas de fatiga
        if fatigue['fatigue_score'] > 70:
            alerts.append({
                'type': 'fatigue',
                'severity': 'high',
                'message': 'Fatiga elevada - Riesgo de microsueños'
            })
        
        return alerts
    
    def _generate_recommendations(self, emotion, stress, fatigue):
        """Genera recomendaciones personalizadas"""
        recommendations = []
        
        # Analizar condición general
        if fatigue['fatigue_score'] > 60 and stress['stress_level'] > 60:
            recommendations.append("Tome un descanso inmediato de 15-20 minutos")
            recommendations.append("Considere cambio de turno si es posible")
        
        elif emotion['dominant_emotion'] in ['tristeza', 'desanimo']:
            recommendations.append("Converse con su supervisor si necesita apoyo")
            recommendations.append("Tome un breve descanso para despejarse")
        
        elif stress['stress_level'] > 70:
            recommendations.extend(stress['recommendations'])
        
        elif fatigue['fatigue_score'] > 50:
            recommendations.extend(fatigue['recommendations'])
        
        else:
            recommendations.append("Continúe con precaución normal")
            recommendations.append("Mantenga hidratación adecuada")
        
        return recommendations[:3]  # Máximo 3 recomendaciones
    
    def _update_operator_profile(self, analysis):
        """Actualiza el perfil del operador"""
        self.operator_profile = {
            'emotional_state': analysis['emotion']['dominant_emotion'],
            'stress_level': analysis['stress']['stress_level'],
            'fatigue_level': analysis['fatigue']['fatigue_score'],
            'overall_condition': analysis['overall_assessment']['status']
        }
    
    def draw_integrated_panel(self, frame):
        """
        Dibuja panel integrado completo con TODOS los análisis.
        
        Args:
            frame: Frame donde dibujar
            
        Returns:
            frame: Frame con panel dibujado
        """
        if self.analysis_history:
            latest = self.analysis_history[-1]
            # Crear instancia temporal de AnalysisDashboard si no existe
            if not hasattr(self, 'dashboard'):
                self.dashboard = AnalysisDashboard()
            # Usar el dashboard mejorado
            return self.dashboard.render(frame, latest)
        
        return frame
    
    def _draw_mini_bar(self, frame, x, y, value, color):
        """Dibuja una barra de progreso pequeña"""
        bar_width = 100
        bar_height = 8
        
        # Fondo
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     (50, 50, 50), -1)
        
        # Progreso
        progress = int(bar_width * (value / 100))
        cv2.rectangle(frame, (x, y), (x + progress, y + bar_height),
                     color, -1)
    
    def get_analysis_summary(self):
        """Obtiene resumen del análisis actual"""
        if not self.analysis_history:
            return None
        
        latest = self.analysis_history[-1]
        return {
            'timestamp': time.time(),
            'operator_condition': self.operator_profile,
            'critical_alerts': [a for a in latest['alerts'] if a['severity'] in ['high', 'critical']],
            'needs_intervention': latest['overall_assessment']['score'] < 40
        }
    
    def reset(self):
        """Reinicia el sistema de análisis"""
        self.emotion_analyzer.reset()
        self.stress_analyzer.reset()
        self.fatigue_monitor.reset()
        self.analysis_history.clear()
        self.operator_profile = {
            'emotional_state': 'neutral',
            'stress_level': 0,
            'fatigue_level': 0,
            'overall_condition': 'good'
        }