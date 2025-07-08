"""
Dashboard de Detección de Bostezos
==================================
Visualización especializada para el módulo de bostezos.
"""

import cv2
import numpy as np
from collections import deque
import time

class YawnDashboard:
    def __init__(self, width=350, position='right'):
        """
        Inicializa el dashboard de bostezos.
        
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
            'yawning': (0, 0, 255),
            'normal': (0, 255, 0),
            'graph_bg': (40, 40, 40),
            'graph_line': (255, 165, 0)
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
        self.mar_history = deque(maxlen=100)
        self.yawn_times = deque(maxlen=10)
        
        # Estadísticas de sesión
        self.session_start = time.time()
        self.total_yawns = 0
        self.longest_yawn = 0
        self.current_operator_start = None
    
    def render(self, frame, analysis_result):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame donde dibujar
            analysis_result: Resultado del análisis
            
        Returns:
            frame: Frame con dashboard
        """
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
        y_offset = self._draw_title(frame, panel_x, y_offset)
        
        # Información del operador
        y_offset = self._draw_operator_info(frame, panel_x, y_offset, analysis_result)
        
        # Estado de calibración
        y_offset = self._draw_calibration_status(frame, panel_x, y_offset, analysis_result)
        
        # Métricas actuales
        y_offset = self._draw_current_metrics(frame, panel_x, y_offset, analysis_result)
        
        # Gráfico MAR
        y_offset = self._draw_mar_graph(frame, panel_x, y_offset)
        
        # Historial de bostezos
        y_offset = self._draw_yawn_history(frame, panel_x, y_offset, analysis_result)
        
        # Estadísticas de sesión
        y_offset = self._draw_session_stats(frame, panel_x, y_offset)
        
        # Alertas
        y_offset = self._draw_alerts(frame, panel_x, y_offset, analysis_result)
        
        return frame
    
    def _draw_title(self, frame, x, y):
        """Dibuja el título del panel"""
        title = "DETECCIÓN DE BOSTEZOS"
        
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
    
    def _draw_operator_info(self, frame, x, y, result):
        """Dibuja información del operador"""
        operator_info = result.get('operator_info')
        
        if operator_info:
            name = operator_info.get('name', 'Desconocido')
            operator_id = operator_info.get('id', 'N/A')
            
            cv2.putText(frame, f"Operador: {name}", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['text_primary'], 1)
            
            y += 20
            cv2.putText(frame, f"ID: {operator_id}", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
        else:
            cv2.putText(frame, "Sin operador", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['text_secondary'], 1)
        
        return y + 25
    
    def _draw_calibration_status(self, frame, x, y, result):
        """Dibuja el estado de calibración"""
        is_calibrated = result.get('is_calibrated', False)
        
        cv2.putText(frame, "CALIBRACIÓN", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_secondary'], 1)
        
        y += 20
        
        if is_calibrated:
            status_text = "PERSONALIZADA"
            status_color = self.colors['success']
            confidence = result.get('calibration_confidence', 0)
            cv2.putText(frame, f"Confianza: {confidence:.0%}", (x + 30, y + 20),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
        else:
            status_text = "POR DEFECTO"
            status_color = self.colors['warning']
        
        cv2.putText(frame, status_text, (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   status_color, 1)
        
        return y + 30
    
    def _draw_current_metrics(self, frame, x, y, result):
        """Dibuja las métricas actuales"""
        detection_result = result.get('detection_result', {})
        
        cv2.putText(frame, "MÉTRICAS", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Estado actual
        is_yawning = detection_result.get('is_yawning', False)
        status = "BOSTEZANDO" if is_yawning else "Normal"
        status_color = self.colors['yawning'] if is_yawning else self.colors['normal']
        
        cv2.putText(frame, f"Estado: {status}", (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   status_color, 1)
        
        y += 25
        
        # Valor MAR
        mar_value = detection_result.get('mar_value', 0)
        mar_threshold = detection_result.get('mar_threshold', 0.7)
        mar_color = self._get_mar_color(mar_value, mar_threshold)
        
        cv2.putText(frame, f"MAR: {mar_value:.3f} / {mar_threshold:.3f}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   mar_color, 1)
        
        # Barra visual de MAR
        y += 5
        bar_width = self.width - 60
        bar_height = 15
        
        cv2.rectangle(frame, (x + 30, y), (x + 30 + bar_width, y + bar_height),
                     self.colors['graph_bg'], -1)
        
        mar_normalized = min(1.0, mar_value)
        value_width = int(bar_width * mar_normalized)
        cv2.rectangle(frame, (x + 30, y), (x + 30 + value_width, y + bar_height),
                     mar_color, -1)
        
        # Línea de umbral
        threshold_x = x + 30 + int(bar_width * mar_threshold)
        cv2.line(frame, (threshold_x, y - 2), (threshold_x, y + bar_height + 2),
                self.colors['danger'], 2)
        
        y += bar_height + 10
        
        # Duración si está bostezando
        if is_yawning:
            duration = detection_result.get('yawn_duration', 0)
            duration_threshold = result.get('duration_threshold', 2.5)
            progress = min(1.0, duration / duration_threshold)
            
            cv2.putText(frame, f"Duración: {duration:.1f}s", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['warning'], 1)
            
            # Barra de progreso
            y += 5
            cv2.rectangle(frame, (x + 30, y), (x + 30 + bar_width, y + 10),
                         self.colors['graph_bg'], -1)
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (x + 30, y), (x + 30 + progress_width, y + 10),
                         self.colors['warning'], -1)
            y += 10
        
        return y + 15
    
    def _draw_mar_graph(self, frame, x, y):
        """Dibuja gráfico histórico de MAR"""
        if len(self.mar_history) < 2:
            return y
        
        cv2.putText(frame, "HISTORIAL MAR", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_secondary'], 1)
        
        y += 20
        
        # Área del gráfico
        graph_width = self.width - 40
        graph_height = 50
        graph_x = x + 20
        graph_y = y
        
        # Fondo
        cv2.rectangle(frame, (graph_x, graph_y), 
                    (graph_x + graph_width, graph_y + graph_height),
                    self.colors['graph_bg'], -1)
        
        # Dibujar datos
        points = []
        for i, (timestamp, mar, threshold) in enumerate(self.mar_history):
            px = graph_x + int(i * graph_width / len(self.mar_history))
            py = graph_y + graph_height - int(min(mar, 1.0) * graph_height)
            points.append((px, py))
        
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], 
                    self.colors['graph_line'], 2)
        
        # Línea de umbral
        if self.mar_history:
            last_threshold = self.mar_history[-1][2]
            threshold_y = graph_y + graph_height - int(last_threshold * graph_height)
            cv2.line(frame, (graph_x, threshold_y), 
                    (graph_x + graph_width, threshold_y),
                    self.colors['danger'], 1)
        
        return y + graph_height + 20
    
    def _draw_yawn_history(self, frame, x, y, result):
        """Dibuja historial de bostezos"""
        yawn_count = result.get('yawn_count', 0)
        window_minutes = result.get('window_minutes', 10)
        max_yawns = result.get('max_yawns', 3)
        
        cv2.putText(frame, f"BOSTEZOS ({window_minutes} min)", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Contador visual
        count_color = self._get_count_color(yawn_count, max_yawns)
        cv2.putText(frame, f"{yawn_count} / {max_yawns}", (x + 30, y),
                   self.fonts['title'], self.font_scales['title'],
                   count_color, 2)
        
        y += 30
        
        # Últimos bostezos
        if self.yawn_times:
            cv2.putText(frame, "Últimos:", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
            y += 20
            
            current_time = time.time()
            for i, yawn_time in enumerate(list(self.yawn_times)[-3:]):
                time_ago = current_time - yawn_time
                minutes = int(time_ago / 60)
                seconds = int(time_ago % 60)
                
                cv2.putText(frame, f"  Hace {minutes}:{seconds:02d}", (x + 40, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['text_primary'], 1)
                y += 15
        
        return y + 10
    
    def _draw_session_stats(self, frame, x, y):
        """Dibuja estadísticas de la sesión"""
        cv2.putText(frame, "SESIÓN", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_secondary'], 1)
        
        y += 20
        
        # Duración
        session_duration = time.time() - self.session_start
        duration_str = f"{int(session_duration // 3600)}:{int((session_duration % 3600) // 60):02d}:{int(session_duration % 60):02d}"
        cv2.putText(frame, f"Duración: {duration_str}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Total de bostezos
        cv2.putText(frame, f"Total bostezos: {self.total_yawns}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Bostezo más largo
        if self.longest_yawn > 0:
            cv2.putText(frame, f"Más largo: {self.longest_yawn:.1f}s", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_primary'], 1)
        
        return y + 25
    
    def _draw_alerts(self, frame, x, y, result):
        """Dibuja alertas activas"""
        if result.get('yawn_count', 0) >= result.get('max_yawns', 3):
            # Alerta de múltiples bostezos (NO fatiga)
            cv2.putText(frame, "⚠ ALERTA BOSTEZOS", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['danger'], 2)
            
            y += 25
            cv2.putText(frame, "Múltiples bostezos detectados", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['danger'], 1)
            
            y += 20
            cv2.putText(frame, "Recomendación: Tomar descanso", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['warning'], 1)
        
        return y + 30
    
    def _update_histories(self, result):
        """Actualiza los historiales"""
        detection_result = result.get('detection_result', {})
        
        # Actualizar historial MAR
        self.mar_history.append((
            time.time(),
            detection_result.get('mar_value', 0),
            detection_result.get('mar_threshold', 0.7)
        ))
        
        # Si se detectó un bostezo completo
        if detection_result.get('yawn_detected', False):
            self.yawn_times.append(time.time())
            self.total_yawns += 1
            
            duration = detection_result.get('yawn_duration', 0)
            if duration > self.longest_yawn:
                self.longest_yawn = duration
    
    def _get_mar_color(self, mar_value, threshold):
        """Obtiene color según valor de MAR"""
        if mar_value > threshold:
            return self.colors['yawning']
        elif mar_value > threshold * 0.8:
            return self.colors['warning']
        else:
            return self.colors['normal']
    
    def _get_count_color(self, count, max_count):
        """Obtiene color según contador"""
        if count >= max_count:
            return self.colors['danger']
        elif count >= max_count * 0.66:
            return self.colors['warning']
        else:
            return self.colors['success']
    
    def reset(self):
        """Reinicia el dashboard"""
        self.mar_history.clear()
        self.yawn_times.clear()
        self.session_start = time.time()
        self.total_yawns = 0
        self.longest_yawn = 0
        self.current_operator_start = None