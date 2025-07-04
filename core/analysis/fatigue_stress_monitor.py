"""
Módulo Monitor de Fatiga y Estrés
=================================
Monitor simplificado que muestra solo niveles de fatiga y estrés.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging

class FatigueStressMonitor:
    def __init__(self):
        """Inicializa el monitor de fatiga y estrés"""
        self.logger = logging.getLogger('FatigueStressMonitor')
        
        # Importar analizador de estrés
        from .stress_analyzer import StressAnalyzer
        self.stress_analyzer = StressAnalyzer()
        
        # Historial para suavizar valores
        self.stress_history = deque(maxlen=10)
        self.fatigue_history = deque(maxlen=10)
        
        # Valores actuales
        self.current_stress = 0
        self.current_fatigue = 0
        
        # Configuración visual
        self.panel_width = 320
        self.panel_height = 200
        self.colors = {
            'background': (0, 0, 0),
            'text': (255, 255, 255),
            'low': (0, 255, 0),      # Verde
            'medium': (0, 165, 255),  # Naranja
            'high': (0, 0, 255),      # Rojo
            'divider': (100, 100, 100)
        }
        
    def update(self, frame, face_landmarks, fatigue_info=None):
        """
        Actualiza los niveles de fatiga y estrés.
        
        Args:
            frame: Frame actual
            face_landmarks: Puntos faciales detectados
            fatigue_info: Información del detector de fatiga principal
            
        Returns:
            dict: Niveles actualizados
        """
        # Actualizar estrés
        stress_result = self.stress_analyzer.analyze(frame, face_landmarks)
        stress_level = stress_result['stress_level']
        self.stress_history.append(stress_level)
        self.current_stress = int(np.mean(self.stress_history))
        
        # Actualizar fatiga
        if fatigue_info:
            # Calcular nivel basado en microsueños
            microsleep_count = fatigue_info.get('microsleep_count', 0)
            
            if microsleep_count >= 3:
                fatigue_level = 80  # Alto
            elif microsleep_count >= 2:
                fatigue_level = 60  # Medio
            elif microsleep_count >= 1:
                fatigue_level = 40  # Bajo
            else:
                fatigue_level = 20  # Normal
            
            self.fatigue_history.append(fatigue_level)
            self.current_fatigue = int(np.mean(self.fatigue_history))
        
        return {
            'stress_level': self.current_stress,
            'fatigue_level': self.current_fatigue,
            'stress_category': self._categorize_level(self.current_stress),
            'fatigue_category': self._categorize_level(self.current_fatigue)
        }
    
    def draw_panel(self, frame, position=(None, None)):
        """
        Dibuja el panel de fatiga y estrés.
        
        Args:
            frame: Frame donde dibujar
            position: (x, y) posición del panel. None = lado derecho
            
        Returns:
            frame: Frame con panel dibujado
        """
        h, w = frame.shape[:2]
        
        # Posición por defecto: lado derecho
        if position[0] is None:
            panel_x = w - self.panel_width - 10
        else:
            panel_x = position[0]
            
        if position[1] is None:
            panel_y = (h - self.panel_height) // 2  # Centrado vertical
        else:
            panel_y = position[1]
        
        # Crear fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (panel_x, panel_y),
                     (panel_x + self.panel_width, panel_y + self.panel_height),
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Marco del panel
        cv2.rectangle(frame,
                     (panel_x, panel_y),
                     (panel_x + self.panel_width, panel_y + self.panel_height),
                     self.colors['divider'], 2)
        
        # Título
        title = "=== INDICADORES VITALES ==="
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_x = panel_x + (self.panel_width - text_size[0]) // 2
        cv2.putText(frame, title, (title_x, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Separador
        cv2.line(frame, 
                (panel_x + 20, panel_y + 45),
                (panel_x + self.panel_width - 20, panel_y + 45),
                self.colors['divider'], 1)
        
        # NIVEL DE ESTRÉS
        y_offset = panel_y + 70
        self._draw_indicator(frame, panel_x + 20, y_offset,
                           "Nivel de Estres", self.current_stress, "😰")
        
        # NIVEL DE FATIGA
        y_offset = panel_y + 130
        self._draw_indicator(frame, panel_x + 20, y_offset,
                           "Nivel de Fatiga", self.current_fatigue, "😴")
        
        return frame
    
    def _draw_indicator(self, frame, x, y, label, value, icon):
        """Dibuja un indicador con barra de progreso"""
        # Icono y etiqueta
        text = f"{icon} {label}:"
        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Valor porcentual
        value_color = self._get_color_for_value(value)
        value_text = f"{value}%"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, value_text, (x + 220, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, value_color, 2)
        
        # Barra de progreso
        bar_y = y + 15
        bar_width = 250
        bar_height = 20
        
        # Fondo de la barra con borde
        cv2.rectangle(frame, (x-1, bar_y-1), 
                     (x + bar_width + 1, bar_y + bar_height + 1),
                     (200, 200, 200), 1)
        cv2.rectangle(frame, (x, bar_y), 
                     (x + bar_width, bar_y + bar_height),
                     (40, 40, 40), -1)
        
        # Barra de progreso con gradiente
        progress_width = int(bar_width * (value / 100))
        
        for i in range(progress_width):
            ratio = i / bar_width
            if ratio < 0.3:  # Verde
                color = (0, 255, 0)
            elif ratio < 0.6:  # Transición verde a naranja
                g = int(255 - (ratio - 0.3) * 255 / 0.3)
                color = (0, g, 255)
            else:  # Naranja a rojo
                r = int((ratio - 0.6) * 255 / 0.4)
                color = (0, 0, 255 - r)
            
            cv2.line(frame, (x + i, bar_y + 1), 
                    (x + i, bar_y + bar_height - 1), color, 1)
        
        # Marcadores de nivel
        markers = [(30, "Bajo"), (60, "Medio"), (80, "Alto")]
        for threshold, marker_label in markers:
            marker_x = x + int(bar_width * (threshold / 100))
            
            # Línea del marcador
            cv2.line(frame, (marker_x, bar_y + bar_height), 
                    (marker_x, bar_y + bar_height + 5),
                    (200, 200, 200), 1)
            
            # Texto del marcador (solo si hay espacio)
            if threshold == 30 or threshold == 80:
                text_size = cv2.getTextSize(marker_label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = marker_x - text_size[0] // 2
                cv2.putText(frame, marker_label, (text_x, bar_y + bar_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    def _get_color_for_value(self, value):
        """Obtiene el color según el valor"""
        if value < 30:
            return self.colors['low']
        elif value < 60:
            return self.colors['medium']
        else:
            return self.colors['high']
    
    def _categorize_level(self, value):
        """Categoriza el nivel"""
        if value < 30:
            return "bajo"
        elif value < 60:
            return "medio"
        elif value < 80:
            return "alto"
        else:
            return "crítico"
    
    def get_status(self):
        """Obtiene el estado actual"""
        return {
            'stress': {
                'level': self.current_stress,
                'category': self._categorize_level(self.current_stress)
            },
            'fatigue': {
                'level': self.current_fatigue,
                'category': self._categorize_level(self.current_fatigue)
            },
            'overall_risk': self._calculate_overall_risk()
        }
    
    def _calculate_overall_risk(self):
        """Calcula el riesgo general basado en ambos indicadores"""
        # Si cualquiera es crítico, riesgo crítico
        if self.current_stress >= 80 or self.current_fatigue >= 80:
            return "crítico"
        # Si ambos son altos, riesgo alto
        elif self.current_stress >= 60 and self.current_fatigue >= 60:
            return "alto"
        # Si alguno es alto, riesgo medio
        elif self.current_stress >= 60 or self.current_fatigue >= 60:
            return "medio"
        # Si ambos son bajos, riesgo bajo
        else:
            return "bajo"
    
    def should_alert(self):
        """Determina si se debe generar una alerta"""
        # Alertar si algún indicador supera el 70%
        return self.current_stress > 70 or self.current_fatigue > 70
    
    def get_recommendations(self):
        """Obtiene recomendaciones basadas en los niveles"""
        recommendations = []
        
        # Recomendaciones por fatiga
        if self.current_fatigue >= 80:
            recommendations.append("⚠️ FATIGA CRÍTICA: Tome un descanso inmediato")
        elif self.current_fatigue >= 60:
            recommendations.append("Fatiga elevada: Considere una pausa de 10 minutos")
        
        # Recomendaciones por estrés
        if self.current_stress >= 80:
            recommendations.append("⚠️ ESTRÉS CRÍTICO: Realice ejercicios de relajación")
        elif self.current_stress >= 60:
            recommendations.append("Estrés alto: Respire profundamente")
        
        # Si ambos están elevados
        if self.current_stress >= 60 and self.current_fatigue >= 60:
            recommendations.insert(0, "⚠️ ALERTA: Fatiga y estrés elevados simultáneamente")
        
        # Si todo está bien
        if not recommendations:
            recommendations.append("✅ Niveles normales - Continúe monitoreando")
        
        return recommendations
    
    def reset(self):
        """Reinicia el monitor"""
        self.stress_history.clear()
        self.fatigue_history.clear()
        self.current_stress = 0
        self.current_fatigue = 0
        self.stress_analyzer.reset()