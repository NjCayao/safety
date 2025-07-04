"""
Módulo Dashboard de Análisis Integrado
=====================================
Panel de visualización completo para todos los análisis.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging

class AnalysisDashboard:
    def __init__(self, panel_width=350, position='right'):
        """
        Inicializa el dashboard de análisis.
        
        Args:
            panel_width: Ancho del panel en píxeles
            position: Posición del panel ('right' o 'left')
        """
        self.logger = logging.getLogger('AnalysisDashboard')
        
        # Configuración del panel
        self.panel_width = panel_width
        self.position = position
        self.margin = 10
        self.section_spacing = 25
        
        # Colores del tema
        self.colors = {
            'background': (0, 0, 0),
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            'accent': (0, 255, 255),
            'success': (0, 255, 0),
            'warning': (0, 165, 255),
            'danger': (0, 0, 255),
            'graph_bg': (50, 50, 50),
            'divider': (100, 100, 100)
        }
        
        # Fuentes
        self.fonts = {
            'title': cv2.FONT_HERSHEY_SIMPLEX,
            'subtitle': cv2.FONT_HERSHEY_SIMPLEX,
            'body': cv2.FONT_HERSHEY_SIMPLEX,
            'small': cv2.FONT_HERSHEY_SIMPLEX
        }
        
        self.font_scales = {
            'title': 0.7,
            'subtitle': 0.6,
            'body': 0.5,
            'small': 0.4
        }
        
        # Historial para gráficos
        self.history_length = 100
        self.emotion_history = deque(maxlen=self.history_length)
        self.stress_history = deque(maxlen=self.history_length)
        self.fatigue_history = deque(maxlen=self.history_length)
        self.pulse_history = deque(maxlen=self.history_length)
        
        # Cache de renderizado
        self.cache_enabled = True
        self.cached_sections = {}
        self.last_update_time = {}
        
        # Estado del modo
        self.is_night_mode = False
        
    def render(self, frame, analysis_data):
        """
        Renderiza el dashboard completo en el frame.
        
        Args:
            frame: Frame donde dibujar
            analysis_data: Diccionario con todos los datos de análisis
            
        Returns:
            frame: Frame con dashboard dibujado
        """
        h, w = frame.shape[:2]
        
        # Determinar posición del panel
        if self.position == 'right':
            panel_x = w - self.panel_width - self.margin
        else:
            panel_x = self.margin
        
        # Crear overlay para el panel
        overlay = frame.copy()
        
        # Fondo del panel con transparencia
        cv2.rectangle(overlay, 
                     (panel_x, self.margin), 
                     (panel_x + self.panel_width, h - self.margin),
                     self.colors['background'], -1)
        
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Renderizar secciones
        y_offset = self.margin + 20
        
        # Título principal
        y_offset = self._draw_main_title(frame, panel_x, y_offset, analysis_data)
        
        # Estado general
        y_offset = self._draw_overall_status(frame, panel_x, y_offset, analysis_data)
        
        # Sección de emociones
        y_offset = self._draw_emotion_section(frame, panel_x, y_offset, 
                                            analysis_data.get('emotion', {}))
        
        # Sección de indicadores vitales
        y_offset = self._draw_vital_indicators(frame, panel_x, y_offset, analysis_data)
        
        # Gráfico de tendencias
        y_offset = self._draw_trend_graph(frame, panel_x, y_offset)
        
        # Alertas activas
        y_offset = self._draw_alerts_section(frame, panel_x, y_offset, 
                                           analysis_data.get('alerts', []))
        
        # Recomendaciones
        y_offset = self._draw_recommendations(frame, panel_x, y_offset,
                                            analysis_data.get('recommendations', []))
        
        # Información del modo
        self._draw_mode_indicator(frame, panel_x, h - 40)
        
        return frame
    
    def _draw_main_title(self, frame, x, y, data):
        """Dibuja el título principal del dashboard"""
        title = "═══ ANÁLISIS INTEGRADO ═══"
        
        # Calcular ancho del texto para centrarlo
        text_size = cv2.getTextSize(title, self.fonts['title'], 
                                   self.font_scales['title'], 2)[0]
        text_x = x + (self.panel_width - text_size[0]) // 2
        
        cv2.putText(frame, title, (text_x, y),
                   self.fonts['title'], self.font_scales['title'],
                   self.colors['text_primary'], 2)
        
        return y + 35
    
    def _draw_overall_status(self, frame, x, y, data):
        """Dibuja el estado general del operador"""
        if 'overall_assessment' in data:
            assessment = data['overall_assessment']
            status = assessment.get('status', 'DESCONOCIDO')
            score = assessment.get('score', 0)
            color = assessment.get('color', self.colors['text_secondary'])
            
            # Título de sección
            cv2.putText(frame, "CONDICIÓN GENERAL", (x + 10, y),
                       self.fonts['subtitle'], self.font_scales['subtitle'],
                       self.colors['text_secondary'], 1)
            
            y += 30
            
            # Estado con color
            cv2.putText(frame, status, (x + 20, y),
                       self.fonts['title'], self.font_scales['title'],
                       color, 2)
            
            # Barra de progreso circular o lineal
            y += 25
            self._draw_score_bar(frame, x + 20, y, score, 
                               self.panel_width - 40, color)
            
            y += 20
        
        # Separador
        cv2.line(frame, (x + 20, y), (x + self.panel_width - 20, y),
                self.colors['divider'], 1)
        
        return y + 15
    
    def _draw_emotion_section(self, frame, x, y, emotion_data):
        """Dibuja la sección de análisis emocional"""
        cv2.putText(frame, "ESTADO EMOCIONAL", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        if emotion_data:
            # Emoción dominante
            dominant = emotion_data.get('dominant_emotion', 'neutral')
            wellbeing = emotion_data.get('wellbeing', 50)
            
            # Emoji mapping
            emotion_emojis = {
                'alegria': '😊', 'tristeza': '😢', 'enojo': '😠',
                'neutral': '😐', 'frustracion': '😤', 'concentracion': '🤔',
                'relajacion': '😌', 'desanimo': '😔'
            }
            
            emotion_text = f"{dominant.upper()}"
            cv2.putText(frame, emotion_text, (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['accent'], 1)
            
            # Mini gráfico de emociones
            y += 25
            emotions = emotion_data.get('emotions', {})
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for emotion, percentage in top_emotions:
                # Nombre de emoción
                cv2.putText(frame, f"{emotion[:8]}:", (x + 20, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['text_secondary'], 1)
                
                # Barra mini
                bar_x = x + 100
                bar_width = 80
                self._draw_mini_bar(frame, bar_x, y - 8, percentage, bar_width)
                
                # Porcentaje
                cv2.putText(frame, f"{percentage}%", (bar_x + bar_width + 5, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['text_secondary'], 1)
                
                y += 18
            
            # Bienestar
            y += 10
            wellbeing_color = self._get_value_color(wellbeing, inverse=True)
            cv2.putText(frame, f"Bienestar: {wellbeing}%", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       wellbeing_color, 1)
            
            # Actualizar historial
            self.emotion_history.append({
                'time': time.time(),
                'emotion': dominant,
                'wellbeing': wellbeing
            })
        
        y += 15
        cv2.line(frame, (x + 20, y), (x + self.panel_width - 20, y),
                self.colors['divider'], 1)
        
        return y + 15
    
    def _draw_vital_indicators(self, frame, x, y, data):
        """Dibuja indicadores vitales (estrés, fatiga, pulso)"""
        cv2.putText(frame, "INDICADORES VITALES", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Grid de indicadores
        indicators = []
        
        # Estrés
        if 'stress' in data:
            stress_level = data['stress']['stress_level']
            indicators.append(('Estrés', stress_level, '%'))
            self.stress_history.append({'time': time.time(), 'value': stress_level})
        
        # Fatiga
        if 'fatigue' in data:
            fatigue_score = data['fatigue']['fatigue_score']
            indicators.append(('Fatiga', fatigue_score, '%'))
            self.fatigue_history.append({'time': time.time(), 'value': fatigue_score})
        
        # Pulso
        if 'pulse' in data and data['pulse'].get('is_valid', False):
            pulse_bpm = data['pulse']['bpm']
            indicators.append(('Pulso', pulse_bpm, 'BPM'))
            self.pulse_history.append({'time': time.time(), 'value': pulse_bpm})
        
        # Anomalías
        if 'anomaly' in data:
            anomaly_score = data['anomaly']['anomaly_score']
            indicators.append(('Anomalías', anomaly_score, '%'))
        
        # Dibujar indicadores en formato grid
        for i, (name, value, unit) in enumerate(indicators):
            row = i // 2
            col = i % 2
            
            ind_x = x + 20 + col * 150
            ind_y = y + row * 50
            
            # Nombre del indicador
            cv2.putText(frame, name, (ind_x, ind_y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
            
            # Valor con color
            value_color = self._get_indicator_color(name, value)
            value_text = f"{int(value)}{unit}"
            cv2.putText(frame, value_text, (ind_x, ind_y + 20),
                       self.fonts['body'], self.font_scales['body'],
                       value_color, 2)
        
        y += len(indicators) * 25 + 20
        cv2.line(frame, (x + 20, y), (x + self.panel_width - 20, y),
                self.colors['divider'], 1)
        
        return y + 15
    
    def _draw_trend_graph(self, frame, x, y):
        """Dibuja gráfico de tendencias"""
        cv2.putText(frame, "TENDENCIAS (1 MIN)", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Área del gráfico
        graph_height = 80
        graph_width = self.panel_width - 40
        graph_x = x + 20
        graph_y = y
        
        # Fondo del gráfico
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height),
                     self.colors['graph_bg'], -1)
        
        # Dibujar líneas de tendencia
        current_time = time.time()
        
        # Estrés (rojo)
        self._draw_trend_line(frame, self.stress_history, graph_x, graph_y,
                             graph_width, graph_height, (0, 0, 255), current_time)
        
        # Fatiga (naranja)
        self._draw_trend_line(frame, self.fatigue_history, graph_x, graph_y,
                             graph_width, graph_height, (0, 165, 255), current_time)
        
        # Bienestar (verde)
        wellbeing_history = [{'time': h['time'], 'value': h['wellbeing']} 
                            for h in self.emotion_history if 'wellbeing' in h]
        self._draw_trend_line(frame, wellbeing_history, graph_x, graph_y,
                             graph_width, graph_height, (0, 255, 0), current_time)
        
        # Leyenda
        legend_y = graph_y + graph_height + 10
        legends = [
            ('Estrés', (0, 0, 255)),
            ('Fatiga', (0, 165, 255)),
            ('Bienestar', (0, 255, 0))
        ]
        
        for i, (label, color) in enumerate(legends):
            legend_x = graph_x + i * 80
            cv2.circle(frame, (legend_x, legend_y), 3, color, -1)
            cv2.putText(frame, label, (legend_x + 8, legend_y + 3),
                       self.fonts['small'], self.font_scales['small'],
                       color, 1)
        
        return legend_y + 20
    
    def _draw_trend_line(self, frame, history, x, y, width, height, color, current_time):
        """Dibuja una línea de tendencia en el gráfico"""
        if len(history) < 2:
            return
        
        # Filtrar datos del último minuto
        recent_data = [h for h in history if current_time - h['time'] < 60]
        
        if len(recent_data) < 2:
            return
        
        # Normalizar puntos
        points = []
        for i, data in enumerate(recent_data):
            # X: posición temporal normalizada
            time_offset = data['time'] - recent_data[0]['time']
            normalized_x = time_offset / 60.0  # Normalizar a 0-1
            px = int(x + normalized_x * width)
            
            # Y: valor normalizado (asumiendo 0-100)
            normalized_y = 1.0 - (data['value'] / 100.0)
            py = int(y + normalized_y * height)
            
            points.append((px, py))
        
        # Dibujar línea
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], color, 2)
    
    def _draw_alerts_section(self, frame, x, y, alerts):
        """Dibuja la sección de alertas activas"""
        if not alerts:
            return y
        
        cv2.putText(frame, "⚠️ ALERTAS ACTIVAS", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['warning'], 1)
        
        y += 25
        
        # Mostrar máximo 3 alertas
        for alert in alerts[:3]:
            severity = alert.get('severity', 'medium')
            message = alert.get('message', '')
            
            # Color según severidad
            if severity == 'critical':
                color = self.colors['danger']
            elif severity == 'high':
                color = self.colors['warning']
            else:
                color = self.colors['text_secondary']
            
            # Icono de alerta
            cv2.circle(frame, (x + 25, y - 3), 3, color, -1)
            
            # Mensaje (dividir si es muy largo)
            if len(message) > 35:
                message = message[:32] + "..."
            
            cv2.putText(frame, message, (x + 35, y),
                       self.fonts['small'], self.font_scales['small'],
                       color, 1)
            
            y += 18
        
        y += 10
        cv2.line(frame, (x + 20, y), (x + self.panel_width - 20, y),
                self.colors['divider'], 1)
        
        return y + 15
    
    def _draw_recommendations(self, frame, x, y, recommendations):
        """Dibuja las recomendaciones"""
        if not recommendations:
            return y
        
        cv2.putText(frame, "💡 RECOMENDACIONES", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_primary'], 1)
        
        y += 25
        
        # Mostrar máximo 3 recomendaciones
        for rec in recommendations[:3]:
            # Dividir texto largo en múltiples líneas
            words = rec.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + word) < 38:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            
            if current_line:
                lines.append(current_line.strip())
            
            # Dibujar cada línea
            for line in lines[:2]:  # Máximo 2 líneas por recomendación
                cv2.putText(frame, f"• {line}", (x + 20, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['text_secondary'], 1)
                y += 15
            
            y += 5
        
        return y
    
    def _draw_mode_indicator(self, frame, x, y):
        """Dibuja indicador del modo actual"""
        mode_text = "🌙 MODO NOCTURNO" if self.is_night_mode else "☀️ MODO DIURNO"
        mode_color = (0, 150, 255) if self.is_night_mode else (255, 200, 0)
        
        cv2.putText(frame, mode_text, (x + 10, y),
                   self.fonts['small'], self.font_scales['small'],
                   mode_color, 1)
    
    def _draw_score_bar(self, frame, x, y, score, width, color):
        """Dibuja una barra de puntuación"""
        height = 10
        
        # Fondo
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['graph_bg'], -1)
        
        # Progreso
        progress_width = int(width * (score / 100))
        cv2.rectangle(frame, (x, y), (x + progress_width, y + height),
                     color, -1)
        
        # Texto del porcentaje
        text = f"{int(score)}%"
        text_size = cv2.getTextSize(text, self.fonts['small'], 
                                   self.font_scales['small'], 1)[0]
        text_x = x + (width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, y + height - 2),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
    
    def _draw_mini_bar(self, frame, x, y, value, width):
        """Dibuja una barra pequeña para visualización"""
        height = 8
        
        # Fondo
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['graph_bg'], -1)
        
        # Progreso
        progress = int(width * (value / 100))
        color = self._get_value_color(value, inverse=True)
        cv2.rectangle(frame, (x, y), (x + progress, y + height),
                     color, -1)
    
    def _get_value_color(self, value, inverse=False):
        """Obtiene color según valor (0-100)"""
        if inverse:
            if value > 70:
                return self.colors['success']
            elif value > 40:
                return self.colors['warning']
            else:
                return self.colors['danger']
        else:
            if value < 30:
                return self.colors['success']
            elif value < 60:
                return self.colors['warning']
            else:
                return self.colors['danger']
    
    def _get_indicator_color(self, indicator_name, value):
        """Obtiene color específico para cada indicador"""
        if indicator_name == 'Estrés':
            return self._get_value_color(value, inverse=False)
        elif indicator_name == 'Fatiga':
            return self._get_value_color(value, inverse=False)
        elif indicator_name == 'Pulso':
            if 60 <= value <= 100:
                return self.colors['success']
            elif 50 <= value < 60 or 100 < value <= 110:
                return self.colors['warning']
            else:
                return self.colors['danger']
        elif indicator_name == 'Anomalías':
            return self._get_value_color(value, inverse=False)
        else:
            return self.colors['text_secondary']
    
    def update_mode(self, is_night_mode):
        """Actualiza el modo día/noche"""
        self.is_night_mode = is_night_mode
        
        # Ajustar colores para modo nocturno
        if is_night_mode:
            self.colors['background'] = (0, 0, 20)
            self.colors['graph_bg'] = (20, 20, 40)
        else:
            self.colors['background'] = (0, 0, 0)
            self.colors['graph_bg'] = (50, 50, 50)
    
    def get_compact_view(self, analysis_data):
        """
        Genera una vista compacta de los datos para logging o display mínimo.
        
        Returns:
            dict: Resumen compacto de métricas clave
        """
        compact = {
            'timestamp': time.time(),
            'overall_status': 'UNKNOWN',
            'key_metrics': {},
            'active_alerts': 0
        }
        
        if 'overall_assessment' in analysis_data:
            compact['overall_status'] = analysis_data['overall_assessment'].get('status', 'UNKNOWN')
        
        # Métricas clave
        if 'emotion' in analysis_data:
            compact['key_metrics']['emotion'] = analysis_data['emotion'].get('dominant_emotion', 'neutral')
            compact['key_metrics']['wellbeing'] = analysis_data['emotion'].get('wellbeing', 0)
        
        if 'stress' in analysis_data:
            compact['key_metrics']['stress'] = analysis_data['stress'].get('stress_level', 0)
        
        if 'fatigue' in analysis_data:
            compact['key_metrics']['fatigue'] = analysis_data['fatigue'].get('fatigue_score', 0)
        
        if 'pulse' in analysis_data and analysis_data['pulse'].get('is_valid'):
            compact['key_metrics']['pulse'] = analysis_data['pulse'].get('bpm', 0)
        
        if 'alerts' in analysis_data:
            compact['active_alerts'] = len(analysis_data['alerts'])
        
        return compact
    
    def export_metrics(self):
        """Exporta métricas históricas para análisis posterior"""
        return {
            'emotion_history': list(self.emotion_history),
            'stress_history': list(self.stress_history),
            'fatigue_history': list(self.fatigue_history),
            'pulse_history': list(self.pulse_history),
            'export_time': time.time()
        }
    
    def reset(self):
        """Reinicia el dashboard"""
        self.emotion_history.clear()
        self.stress_history.clear()
        self.fatigue_history.clear()
        self.pulse_history.clear()
        self.cached_sections.clear()
        self.last_update_time.clear()