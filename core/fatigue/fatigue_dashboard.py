"""
Dashboard de Fatiga
==================
Visualización especializada para el módulo de fatiga.
"""

import cv2
import numpy as np
from collections import deque
import time

class FatigueDashboard:
    def __init__(self, width=350, position='right'):
        """
        Inicializa el dashboard de fatiga.
        
        Args:
            width: Ancho del panel
            position: Posición ('left' o 'right')
        """
        self.width = width
        self.position = position
        self.margin = 10
        
        # Colores del tema
        self.colors = {
            'background': (20, 20, 20),
            'text_primary': (255, 255, 255),
            'text_secondary': (180, 180, 180),
            'success': (0, 255, 0),
            'warning': (0, 165, 255),
            'danger': (0, 0, 255),
            'accent': (255, 255, 0),
            'graph_bg': (40, 40, 40),
            'graph_line': (0, 255, 255)
        }
        
        # Fuentes
        self.fonts = {
            'title': cv2.FONT_HERSHEY_SIMPLEX,
            'body': cv2.FONT_HERSHEY_SIMPLEX,
            'small': cv2.FONT_HERSHEY_SIMPLEX
        }
        
        self.font_scales = {
            'title': 0.7,
            'body': 0.5,
            'small': 0.4
        }
        
        # Historial para gráficos
        self.ear_history = deque(maxlen=100)
        self.fatigue_history = deque(maxlen=100)
        self.microsleep_history = deque(maxlen=50)
        
        # Cache de elementos estáticos
        self.static_elements_cache = None
        self.last_cache_update = 0
        
    def render(self, frame, analysis_result):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame donde dibujar
            analysis_result: Resultado del análisis de fatiga
            
        Returns:
            frame: Frame con dashboard
        """
        if not analysis_result:
            return frame
        
        h, w = frame.shape[:2]
        
        # Determinar posición X del panel
        if self.position == 'right':
            panel_x = w - self.width - self.margin
        else:
            panel_x = self.margin
        
        # Crear overlay para el panel
        overlay = frame.copy()
        
        # Fondo del panel
        cv2.rectangle(overlay, 
                     (panel_x, self.margin), 
                     (panel_x + self.width, h - self.margin),
                     self.colors['background'], -1)
        
        # Aplicar transparencia
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Actualizar historiales
        self._update_histories(analysis_result)
        
        # Renderizar secciones
        y_offset = self.margin + 20
        
        # Título
        y_offset = self._draw_title(frame, panel_x, y_offset, analysis_result)
        
        # Estado del operador
        y_offset = self._draw_operator_status(frame, panel_x, y_offset, analysis_result)
        
        # Métricas principales
        y_offset = self._draw_main_metrics(frame, panel_x, y_offset, analysis_result)
        
        # Gráfico EAR
        y_offset = self._draw_ear_graph(frame, panel_x, y_offset)
        
        # Estadísticas
        y_offset = self._draw_statistics(frame, panel_x, y_offset, analysis_result)
        
        # Alertas
        y_offset = self._draw_alerts(frame, panel_x, y_offset, analysis_result)
        
        # Indicadores visuales en el video principal
        self._draw_video_overlays(frame, analysis_result)
        
        return frame
    
    def _draw_title(self, frame, x, y, result):
        """Dibuja el título del panel"""
        title = "MONITOR DE FATIGA"
        
        # Centrar título
        text_size = cv2.getTextSize(title, self.fonts['title'], 
                                   self.font_scales['title'], 2)[0]
        text_x = x + (self.width - text_size[0]) // 2
        
        cv2.putText(frame, title, (text_x, y),
                   self.fonts['title'], self.font_scales['title'],
                   self.colors['accent'], 2)
        
        # Línea separadora
        y += 10
        cv2.line(frame, (x + 20, y), (x + self.width - 20, y),
                self.colors['text_secondary'], 1)
        
        return y + 20
    
    def _draw_operator_status(self, frame, x, y, result):
        """Dibuja información del operador"""
        # Nombre del operador
        operator_name = result.get('operator_name', 'No identificado')
        cv2.putText(frame, f"Operador: {operator_name}", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_primary'], 1)
        
        y += 25
        
        # Estado de calibración
        is_calibrated = result.get('is_calibrated', False)
        calib_text = "Calibrado" if is_calibrated else "Sin calibrar"
        calib_color = self.colors['success'] if is_calibrated else self.colors['warning']
        
        cv2.putText(frame, f"Estado: {calib_text}", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   calib_color, 1)
        
        return y + 30
    
    def _draw_main_metrics(self, frame, x, y, result):
        """Dibuja las métricas principales"""
        # Título de sección
        cv2.putText(frame, "METRICAS ACTUALES", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # EAR actual
        ear_value = result.get('ear_value', 0)
        ear_threshold = result.get('ear_threshold', 0.25)
        ear_color = self._get_ear_color(ear_value, ear_threshold)
        
        cv2.putText(frame, f"EAR: {ear_value:.3f}", (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   ear_color, 1)
        
        # Barra visual de EAR
        bar_width = self.width - 60
        bar_height = 15
        y += 5
        
        # Fondo de la barra
        cv2.rectangle(frame, (x + 30, y), (x + 30 + bar_width, y + bar_height),
                     self.colors['graph_bg'], -1)
        
        # Valor actual
        ear_normalized = min(1.0, ear_value / 0.4)  # Normalizar a 0-0.4
        value_width = int(bar_width * ear_normalized)
        cv2.rectangle(frame, (x + 30, y), (x + 30 + value_width, y + bar_height),
                     ear_color, -1)
        
        # Línea de umbral
        threshold_x = x + 30 + int(bar_width * (ear_threshold / 0.4))
        cv2.line(frame, (threshold_x, y - 2), (threshold_x, y + bar_height + 2),
                self.colors['danger'], 2)
        
        y += bar_height + 15
        
        # Porcentaje de fatiga
        fatigue_pct = result.get('fatigue_percentage', 0)
        fatigue_color = self._get_fatigue_color(fatigue_pct)
        
        cv2.putText(frame, f"Nivel de Fatiga: {fatigue_pct}%", (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   fatigue_color, 1)
        
        # Barra de fatiga
        y += 5
        cv2.rectangle(frame, (x + 30, y), (x + 30 + bar_width, y + bar_height),
                     self.colors['graph_bg'], -1)
        
        fatigue_width = int(bar_width * (fatigue_pct / 100))
        cv2.rectangle(frame, (x + 30, y), (x + 30 + fatigue_width, y + bar_height),
                     fatigue_color, -1)
        
        y += bar_height + 20
        
        # Estado de ojos
        eyes_closed = result.get('eyes_closed', False)
        eyes_duration = result.get('eyes_closed_duration', 0)
        
        if eyes_closed and eyes_duration > 0:
            cv2.putText(frame, f"Ojos cerrados: {eyes_duration:.1f}s", (x + 30, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['danger'], 1)
            y += 25
        
        return y + 10
    
    def _draw_ear_graph(self, frame, x, y):
        """Dibuja gráfico histórico de EAR"""
        if len(self.ear_history) < 2:
            return y
        
        # Título
        cv2.putText(frame, "HISTORIAL EAR (30s)", (x + 20, y),
                self.fonts['small'], self.font_scales['small'],
                self.colors['text_secondary'], 1)
        
        y += 20
        
        # Área del gráfico
        graph_width = self.width - 40
        graph_height = 60
        graph_x = x + 20
        graph_y = y
        
        # Fondo del gráfico
        cv2.rectangle(frame, (graph_x, graph_y), 
                    (graph_x + graph_width, graph_y + graph_height),
                    self.colors['graph_bg'], -1)
        
        # Dibujar línea de EAR
        points = []
        # CAMBIO AQUÍ: Desempaquetar 3 valores en lugar de 2
        for i, data in enumerate(self.ear_history):
            # Si es una tupla de 3 valores
            if isinstance(data, tuple) and len(data) >= 2:
                timestamp = data[0]
                ear = data[1]
            else:
                # Por compatibilidad
                timestamp, ear = data[0], data[1]
                
            # Posición X basada en índice
            px = graph_x + int(i * graph_width / len(self.ear_history))
            
            # Posición Y (invertida, normalizada 0-0.4)
            normalized_y = 1.0 - min(1.0, ear / 0.4)
            py = graph_y + int(normalized_y * graph_height)
            
            points.append((px, py))
        
        # Dibujar línea
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], self.colors['graph_line'], 2)
        
        # Línea de umbral promedio
        if self.ear_history:
            # Obtener el umbral del último registro si existe
            if isinstance(self.ear_history[-1], tuple) and len(self.ear_history[-1]) > 2:
                avg_threshold = self.ear_history[-1][2]
            else:
                avg_threshold = 0.25
                
            threshold_y = graph_y + int((1.0 - avg_threshold / 0.4) * graph_height)
            cv2.line(frame, (graph_x, threshold_y), 
                    (graph_x + graph_width, threshold_y),
                    self.colors['danger'], 1)
        
        return graph_y + graph_height + 20
    
    def _draw_statistics(self, frame, x, y, result):
        """Dibuja estadísticas de la sesión"""
        cv2.putText(frame, "ESTADISTICAS", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Microsueños
        microsleep_count = result.get('microsleep_count', 0)
        total_microsleeps = result.get('total_microsleeps', 0)
        
        ms_color = self.colors['danger'] if microsleep_count >= 3 else self.colors['warning'] if microsleep_count >= 1 else self.colors['text_primary']
        
        cv2.putText(frame, f"Microsuenos (10min): {microsleep_count}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   ms_color, 1)
        
        y += 20
        
        cv2.putText(frame, f"Total sesion: {total_microsleeps}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Frecuencia de parpadeo
        blink_rate = result.get('blink_rate', 0)
        blink_color = self.colors['warning'] if blink_rate > 30 or blink_rate < 10 else self.colors['text_primary']
        
        cv2.putText(frame, f"Parpadeos/min: {int(blink_rate)}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   blink_color, 1)
        
        return y + 30
    
    def _draw_alerts(self, frame, x, y, result):
        """Dibuja alertas activas"""
        alerts = result.get('alerts', [])
        
        if not alerts:
            return y
        
        cv2.putText(frame, "ALERTAS", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['danger'], 1)
        
        y += 25
        
        for alert in alerts[:3]:  # Máximo 3 alertas
            severity = alert.get('severity', 'medium')
            message = alert.get('message', '')
            
            color = self.colors['danger'] if severity == 'critical' else self.colors['warning']
            
            # Icono de alerta
            cv2.putText(frame, "!", (x + 30, y),
                       self.fonts['body'], self.font_scales['body'],
                       color, 2)
            
            # Mensaje (truncar si es muy largo)
            if len(message) > 30:
                message = message[:27] + "..."
            
            cv2.putText(frame, message, (x + 50, y),
                       self.fonts['small'], self.font_scales['small'],
                       color, 1)
            
            y += 20
        
        return y + 10
    
    def _draw_video_overlays(self, frame, result):
        """Dibuja overlays en el video principal"""
        # Si hay fatiga crítica, agregar borde rojo
        if result.get('is_critical', False):
            h, w = frame.shape[:2]
            thickness = 5
            cv2.rectangle(frame, (thickness, thickness), 
                         (w - thickness, h - thickness),
                         self.colors['danger'], thickness)
        
        # Si los ojos están cerrados, mostrar timer
        if result.get('eyes_closed', False):
            duration = result.get('eyes_closed_duration', 0)
            if duration > 0.5:
                text = f"OJOS CERRADOS: {duration:.1f}s"
                text_size = cv2.getTextSize(text, self.fonts['title'], 
                                           self.font_scales['title'], 2)[0]
                
                x = (frame.shape[1] - text_size[0]) // 2
                y = 100
                
                # Fondo para el texto
                cv2.rectangle(frame, (x - 10, y - 30), 
                             (x + text_size[0] + 10, y + 10),
                             self.colors['danger'], -1)
                
                cv2.putText(frame, text, (x, y),
                           self.fonts['title'], self.font_scales['title'],
                           self.colors['text_primary'], 2)
    
    def _update_histories(self, result):
        """Actualiza los historiales para gráficos"""
        timestamp = result.get('timestamp', time.time())
        
        # EAR
        self.ear_history.append((
            timestamp,
            result.get('ear_value', 0),
            result.get('ear_threshold', 0.25)
        ))
        
        # Fatiga
        self.fatigue_history.append((
            timestamp,
            result.get('fatigue_percentage', 0)
        ))
        
        # Microsueños
        if result.get('microsleep_detected', False):
            self.microsleep_history.append(timestamp)
    
    def _get_ear_color(self, ear_value, threshold):
        """Obtiene color según valor de EAR"""
        if ear_value < threshold * 0.8:
            return self.colors['danger']
        elif ear_value < threshold:
            return self.colors['warning']
        else:
            return self.colors['success']
    
    def _get_fatigue_color(self, percentage):
        """Obtiene color según porcentaje de fatiga"""
        if percentage >= 80:
            return self.colors['danger']
        elif percentage >= 60:
            return self.colors['warning']
        elif percentage >= 40:
            return self.colors['text_primary']
        else:
            return self.colors['success']
    
    def reset(self):
        """Reinicia el dashboard"""
        self.ear_history.clear()
        self.fatigue_history.clear()
        self.microsleep_history.clear()
        self.static_elements_cache = None