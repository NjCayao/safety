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
    def __init__(self, panel_width=300, position='right'):
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
        self.section_spacing = 15
        
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
        
        # Fuentes - AUMENTADAS PARA MEJOR LEGIBILIDAD
        self.fonts = {
            'title': cv2.FONT_HERSHEY_SIMPLEX,
            'subtitle': cv2.FONT_HERSHEY_SIMPLEX,
            'body': cv2.FONT_HERSHEY_SIMPLEX,
            'small': cv2.FONT_HERSHEY_SIMPLEX
        }
        
        self.font_scales = {
            'title': 0.5,      # Aumentado de 0.7
            'subtitle': 0.45,   # Aumentado de 0.6
            'body': 0.4,      # Aumentado de 0.5
            'small': 0.35      # Aumentado de 0.4
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
        self._current_anomaly_data = None
        
        # Diccionario de traducción de emociones
        self.emotion_spanish = {
            'neutral': 'neutral',
            'happy': 'feliz',
            'sad': 'triste',
            'angry': 'enojado',
            'surprised': 'sorprendido',
            'fear': 'miedo',
            'disgust': 'disgusto'
        }
        
    def render(self, frame, analysis_data):
        """
        Renderiza el dashboard completo en el frame.
        
        Args:
            frame: Frame donde dibujar
            analysis_data: Diccionario con todos los datos de análisis
            
        Returns:
            frame: Frame con dashboard dibujado
        """
        # Guardar datos de anomalías para usar en alertas
        self._current_anomaly_data = analysis_data.get('analysis', {}).get('anomaly', {})
        
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
                                            analysis_data.get('analysis', {}).get('emotion', {}))
        
        # Sección de indicadores vitales
        y_offset = self._draw_vital_indicators(frame, panel_x, y_offset, analysis_data.get('analysis', {}))
        
        # Sección de anomalías
        y_offset = self._draw_anomaly_section(frame, panel_x, y_offset, 
                                            analysis_data.get('analysis', {}).get('anomaly', {}))
        
        # Gráfico de tendencias
        # y_offset = self._draw_trend_graph(frame, panel_x, y_offset)
        
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
        title = "SISTEMA DE ANALISIS INTEGRADO"
        
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
            cv2.putText(frame, "CONDICION GENERAL", (x + 10, y),
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
            # Obtener datos necesarios
            dominant = emotion_data.get('dominant_emotion', 'neutral')
            wellbeing = emotion_data.get('wellbeing', 50)
            
            # Mostrar emoción dominante
            emotion_esp = self.emotion_spanish.get(dominant, dominant)
            emotion_text = f"{emotion_esp.upper()}"
            cv2.putText(frame, emotion_text, (x + 20, y),
                    self.fonts['body'], self.font_scales['body'],
                    self.colors['accent'], 1)
            
            # Mostrar TODAS las emociones
            y += 25
            emotions = emotion_data.get('emotions', {})
            
            # Ordenar por valor descendente
            all_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            for emotion, percentage in all_emotions:
                # Traducir nombre de emoción
                emotion_esp = self.emotion_spanish.get(emotion, emotion)
                
                # Color según el porcentaje
                if percentage > 30:
                    color = self.colors['accent']
                elif percentage > 15:
                    color = self.colors['text_primary']
                else:
                    color = self.colors['text_secondary']
                
                # Nombre de emoción
                cv2.putText(frame, f"{emotion_esp}:", (x + 20, y),
                        self.fonts['small'], self.font_scales['small'],
                        color, 1)
                
                # Barra mini
                bar_x = x + 80
                bar_width = 60
                self._draw_mini_bar(frame, bar_x, y - 6, percentage, bar_width)
                
                # Porcentaje
                cv2.putText(frame, f"{percentage}%", (bar_x + bar_width + 5, y),
                        self.fonts['small'], self.font_scales['small'],
                        color, 1)
                
                y += 15
            
            # Bienestar (FUERA del bucle)
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
        """Dibuja indicadores vitales (estres, fatiga, pulso)"""
        cv2.putText(frame, "INDICADORES VITALES", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Grid de indicadores
        indicators = []
        
        # Estrés
        if 'stress' in data:
            stress_level = data['stress']['stress_level']
            indicators.append(('Estres', stress_level, '%'))
            self.stress_history.append({'time': time.time(), 'value': stress_level})
        
        # Fatiga
        if 'fatigue' in data:
            fatigue_score = data['fatigue']['fatigue_percentage']
            indicators.append(('Fatiga', fatigue_score, '%'))
            self.fatigue_history.append({'time': time.time(), 'value': fatigue_score})
        
        # Pulso
        if 'pulse' in data:
            if data['pulse'].get('is_valid', False):
                pulse_bpm = data['pulse']['bpm']
                indicators.append(('Pulso', pulse_bpm, 'LPM'))
                self.pulse_history.append({'time': time.time(), 'value': pulse_bpm})
            else:
                # Mostrar que está midiendo
                indicators.append(('Pulso', 0, 'Midiendo...'))
        
        
        # Anomalías (ahora incluido en vital indicators)
        
        # Dibujar indicadores en formato grid
        for i, (name, value, unit) in enumerate(indicators):
            row = i // 2
            col = i % 2
            
            ind_x = x + 20 + col * 180
            ind_y = y + row * 35
            
            # Nombre del indicador
            cv2.putText(frame, name, (ind_x, ind_y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
            
            # Valor con color
            value_color = self._get_indicator_color(name, value)
            value_text = f"{int(value)}{unit}"
            cv2.putText(frame, value_text, (ind_x, ind_y + 20),
                       self.fonts['body'], self.font_scales['body'],
                       value_color, 1)
        
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
        # Recolectar todas las alertas (incluyendo anomalías)
        all_alerts = list(alerts) if alerts else []
        
        # Si hay datos de anomalías con alertas, agregarlas
        if hasattr(self, '_current_anomaly_data') and self._current_anomaly_data:
            if 'alerts' in self._current_anomaly_data:
                all_alerts.extend(self._current_anomaly_data['alerts'])
        
        if not all_alerts:
            return y
        
        cv2.putText(frame, "ALERTAS ACTIVAS", (x + 10, y),
                   self.fonts['subtitle'], self.font_scales['subtitle'],
                   self.colors['warning'], 1)
        
        y += 25
        
        # Mostrar máximo 4 alertas
        for alert in all_alerts[:4]:
            severity = alert.get('severity', alert.get('level', 'medium'))
            message = alert.get('message', '')
            
            # Color según severidad
            if severity in ['critical', 'CRITICAL']:
                color = self.colors['danger']
            elif severity in ['high', 'HIGH']:
                color = self.colors['warning']
            else:
                color = self.colors['text_secondary']
            
            # Punto de alerta en lugar de círculo
            cv2.putText(frame, "[!]", (x + 20, y),
                       self.fonts['small'], self.font_scales['small'],
                       color, 1)
            
            # Mensaje (dividir si es muy largo)
            if len(message) > 35:
                message = message[:32] + "..."
            
            cv2.putText(frame, message, (x + 40, y),
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
        
        cv2.putText(frame, "RECOMENDACIONES", (x + 10, y),
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
    
    def _draw_anomaly_section(self, frame, x, y, anomaly_data):
        """Dibuja la sección de detección de anomalías"""
        cv2.putText(frame, "DETECCION DE ANOMALIAS", (x + 10, y),
                self.fonts['subtitle'], self.font_scales['subtitle'],
                self.colors['text_secondary'], 1)
        
        y += 25
        
        if anomaly_data and 'indicators' in anomaly_data:
            indicators = anomaly_data['indicators']
            
            # Intoxicación
            if 'intoxication' in indicators:
                intox = indicators['intoxication']
                level = intox['level']
                status = intox['status']
                color = self._get_value_color(level, inverse=False)
                
                # Texto primero
                cv2.putText(frame, f"Intoxicacion: {level}% [{status}]", (x + 20, y),
                        self.fonts['body'], self.font_scales['body'],
                        color, 1)
                
                # Barra DEBAJO del texto
                y += 15
                self._draw_mini_bar(frame, x + 20, y, level, 200)
                y += 20
            
            # Riesgo neurológico
            if 'neurological' in indicators:
                neuro = indicators['neurological']
                level = neuro['level']
                status = neuro['status']
                color = self._get_value_color(level, inverse=False)
                
                # Texto primero
                cv2.putText(frame, f"Riesgo Neurol.: {level}% [{status}]", (x + 20, y),
                        self.fonts['body'], self.font_scales['body'],
                        color, 1)
                
                # Barra DEBAJO del texto
                y += 15
                self._draw_mini_bar(frame, x + 20, y, level, 200)
                y += 20
            
            # Comportamiento errático
            if 'erratic' in indicators:
                erratic = indicators['erratic']
                level = erratic['level']
                status = erratic['status']
                color = self._get_value_color(level, inverse=False)
                
                # Texto primero
                cv2.putText(frame, f"Comp. Erratico: {level}% [{status}]", (x + 20, y),
                        self.fonts['body'], self.font_scales['body'],
                        color, 1)
                
                # Barra DEBAJO del texto
                y += 15
                self._draw_mini_bar(frame, x + 20, y, level, 200)
                y += 15
        
        y += 10
        cv2.line(frame, (x + 20, y), (x + self.panel_width - 20, y),
                self.colors['divider'], 1)
        
        return y + 15
    
    def draw_calibration_progress(self, frame, progress, operator_name):
        """Dibuja el progreso de calibración"""
        h, w = frame.shape[:2]
        
        # Crear overlay oscuro
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Panel central
        panel_width = 400
        panel_height = 200
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        # Fondo del panel
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (100, 100, 100), 2)
        
        # Título
        title = "CALIBRACIÓN EN PROGRESO"
        text_size = cv2.getTextSize(title, self.fonts['title'], 
                                   self.font_scales['title'], 2)[0]
        title_x = panel_x + (panel_width - text_size[0]) // 2
        cv2.putText(frame, title, (title_x, panel_y + 40),
                   self.fonts['title'], self.font_scales['title'],
                   self.colors['accent'], 2)
        
        # Nombre del operador
        op_text = f"Operador: {operator_name}"
        text_size = cv2.getTextSize(op_text, self.fonts['body'], 
                                   self.font_scales['body'], 1)[0]
        op_x = panel_x + (panel_width - text_size[0]) // 2
        cv2.putText(frame, op_text, (op_x, panel_y + 70),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_primary'], 1)
        
        # Barra de progreso
        bar_x = panel_x + 50
        bar_y = panel_y + 100
        bar_width = panel_width - 100
        bar_height = 30
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Progreso
        progress_width = int(bar_width * (progress / 100))
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     self.colors['accent'], -1)
        
        # Porcentaje
        percent_text = f"{progress}%"
        text_size = cv2.getTextSize(percent_text, self.fonts['body'],
                                   self.font_scales['body'], 2)[0]
        percent_x = bar_x + (bar_width - text_size[0]) // 2
        cv2.putText(frame, percent_text, (percent_x, bar_y + bar_height - 8),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_primary'], 2)
        
        # Instrucciones
        instruction = "Por favor, mantenga expresión neutral"
        text_size = cv2.getTextSize(instruction, self.fonts['small'],
                                   self.font_scales['small'], 1)[0]
        inst_x = panel_x + (panel_width - text_size[0]) // 2
        cv2.putText(frame, instruction, (inst_x, panel_y + 160),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_secondary'], 1)
        
        return frame
                
        
    def _draw_mode_indicator(self, frame, x, y):
        """Dibuja indicador del modo actual"""
        mode_text = "MODO NOCTURNO" if self.is_night_mode else "MODO DIURNO"
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
    
    def get_config(self):
        """Obtiene la configuración actual del dashboard"""
        return {
            'panel_width': self.panel_width,
            'position': self.position,
            'cache_enabled': self.cache_enabled,
            'history_length': self.history_length,
            'is_night_mode': self.is_night_mode
        }
    
    def update_config(self, config):
        """Actualiza la configuración del dashboard"""
        if 'panel_width' in config:
            self.panel_width = config['panel_width']
        if 'position' in config:
            self.position = config['position']
        if 'cache_enabled' in config:
            self.cache_enabled = config['cache_enabled']
        if 'history_length' in config:
            self.history_length = config['history_length']
            # Redimensionar historiales
            self._resize_histories(config['history_length'])
    
    def _resize_histories(self, new_length):
        """Redimensiona los historiales de datos"""
        self.history_length = new_length
        # Crear nuevos deques con el nuevo tamaño
        self.emotion_history = deque(self.emotion_history, maxlen=new_length)
        self.stress_history = deque(self.stress_history, maxlen=new_length)
        self.fatigue_history = deque(self.fatigue_history, maxlen=new_length)
        self.pulse_history = deque(self.pulse_history, maxlen=new_length)
    
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