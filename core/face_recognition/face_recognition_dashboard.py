"""
Dashboard de Reconocimiento Facial (SIMPLIFICADO)
=================================================
Visualización enfocada solo en reconocimiento.
"""

import cv2
import numpy as np
from collections import deque
import time

class FaceRecognitionDashboard:
    def __init__(self, width=350, position='right'):
        """
        Inicializa el dashboard de reconocimiento facial.
        
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
            'registered': (0, 255, 0),
            'unknown': (0, 0, 255),
            'graph_bg': (40, 40, 40),
            'graph_confidence': (0, 255, 0)
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
        self.confidence_history = deque(maxlen=60)  # Últimos 60 frames
        
        # Estadísticas de sesión
        self.session_start = time.time()
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.unknown_detections = 0
        self.operators_seen = set()
        self.current_operator_start = None
        self.current_operator_id = None
        
        # Tiempo de operador desconocido
        self.unknown_start_time = None
    
    def render(self, frame, recognition_result):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame donde dibujar
            recognition_result: Resultado del reconocimiento
            
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
        
        # Actualizar estadísticas
        self._update_statistics(recognition_result)
        
        # Renderizar secciones
        y_offset = self.margin + 20
        
        # Título
        y_offset = self._draw_title(frame, panel_x, y_offset)
        
        # Información del operador actual
        y_offset = self._draw_operator_info(frame, panel_x, y_offset, recognition_result)
        
        # Estado de calibración
        y_offset = self._draw_calibration_status(frame, panel_x, y_offset, recognition_result)
        
        # Alerta de operador desconocido (si aplica)
        y_offset = self._draw_unknown_operator_alert(frame, panel_x, y_offset, recognition_result)
        
        # Gráfico de confianza
        y_offset = self._draw_confidence_graph(frame, panel_x, y_offset)
        
        # Estadísticas de sesión
        y_offset = self._draw_session_stats(frame, panel_x, y_offset)
        
        return frame
    
    def _draw_title(self, frame, x, y):
        """Dibuja el título del panel"""
        title = "RECONOCIMIENTO FACIAL"
        
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
        """Dibuja información del operador actual"""
        operator_info = result.get('operator_info')
        
        cv2.putText(frame, "OPERADOR ACTUAL", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        if operator_info:
            # Verificar si está registrado
            is_registered = operator_info.get('is_registered', False)
            
            if is_registered:
                # Operador registrado
                name = operator_info.get('name', 'Desconocido')
                operator_id = operator_info.get('id', 'N/A')
                confidence = operator_info.get('confidence', 0)
                
                # Nombre
                cv2.putText(frame, name, (x + 30, y),
                           self.fonts['body'], self.font_scales['body'],
                           self.colors['registered'], 2)
                
                y += 25
                
                # ID
                cv2.putText(frame, f"ID: {operator_id}", (x + 30, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['text_secondary'], 1)
                
                y += 20
                
                # Confianza
                conf_color = self._get_confidence_color(confidence)
                cv2.putText(frame, f"Confianza: {confidence:.2f}", (x + 30, y),
                           self.fonts['small'], self.font_scales['small'],
                           conf_color, 1)
                
                y += 20
                
                # Tiempo en el puesto
                if self.current_operator_id == operator_id and self.current_operator_start:
                    duration = time.time() - self.current_operator_start
                    duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
                    cv2.putText(frame, f"Tiempo: {duration_str}", (x + 30, y),
                               self.fonts['small'], self.font_scales['small'],
                               self.colors['text_primary'], 1)
            else:
                # Operador NO registrado
                cv2.putText(frame, "NO REGISTRADO", (x + 30, y),
                           self.fonts['body'], self.font_scales['body'],
                           self.colors['unknown'], 2)
                
                y += 25
                
                cv2.putText(frame, "Operador desconocido", (x + 30, y),
                           self.fonts['small'], self.font_scales['small'],
                           self.colors['unknown'], 1)
        else:
            # No hay operador
            cv2.putText(frame, "Sin deteccion", (x + 30, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['text_secondary'], 1)
        
        return y + 30
    
    def _draw_calibration_status(self, frame, x, y, result):
        """Dibuja el estado de calibración"""
        is_calibrated = result.get('is_calibrated', False)
        
        if result.get('operator_info') and result['operator_info'].get('is_registered', False):
            cv2.putText(frame, "CALIBRACION", (x + 20, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
            
            y += 20
            
            if is_calibrated:
                status_text = "PERSONALIZADA"
                status_color = self.colors['success']
            else:
                status_text = "POR DEFECTO"
                status_color = self.colors['warning']
            
            cv2.putText(frame, status_text, (x + 30, y),
                       self.fonts['body'], self.font_scales['body'],
                       status_color, 1)
            
            y += 25
        
        return y
    
    def _draw_unknown_operator_alert(self, frame, x, y, result):
        """Dibuja alerta de operador desconocido si aplica"""
        operator_info = result.get('operator_info')
        
        if operator_info and not operator_info.get('is_registered', False):
            # Calcular tiempo transcurrido
            if self.unknown_start_time:
                elapsed = time.time() - self.unknown_start_time
                minutes = int(elapsed / 60)
                seconds = int(elapsed % 60)
                
                # Color según tiempo
                if minutes >= 10:
                    alert_color = self.colors['danger']
                elif minutes >= 5:
                    alert_color = self.colors['warning']
                else:
                    alert_color = self.colors['text_secondary']
                
                # Dibujar alerta
                cv2.putText(frame, "⚠ ALERTA SEGURIDAD", (x + 20, y),
                           self.fonts['body'], self.font_scales['body'],
                           alert_color, 2)
                
                y += 25
                
                cv2.putText(frame, f"Desconocido por: {minutes}:{seconds:02d}", (x + 30, y),
                           self.fonts['small'], self.font_scales['small'],
                           alert_color, 1)
                
                y += 20
                
                # Mostrar tiempo restante para reporte
                remaining = max(0, (15 * 60) - elapsed)
                if remaining > 0:
                    rem_min = int(remaining / 60)
                    rem_sec = int(remaining % 60)
                    cv2.putText(frame, f"Reporte en: {rem_min}:{rem_sec:02d}", (x + 30, y),
                               self.fonts['small'], self.font_scales['small'],
                               self.colors['text_secondary'], 1)
                else:
                    cv2.putText(frame, "REPORTE ENVIADO", (x + 30, y),
                               self.fonts['small'], self.font_scales['small'],
                               self.colors['danger'], 2)
                
                y += 25
        
        return y
    
    def _draw_confidence_graph(self, frame, x, y):
        """Dibuja gráfico de confianza"""
        if len(self.confidence_history) < 2:
            return y
        
        cv2.putText(frame, "CONFIANZA", (x + 20, y),
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
        
        # Líneas de referencia
        for ref_val in [0.5, 0.7]:
            ref_y = graph_y + graph_height - int(ref_val * graph_height)
            cv2.line(frame, (graph_x, ref_y), (graph_x + graph_width, ref_y),
                    self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        # Dibujar datos
        if self.confidence_history:
            points = []
            for i, value in enumerate(self.confidence_history):
                px = graph_x + int(i * graph_width / len(self.confidence_history))
                py = graph_y + graph_height - int(value * graph_height)
                points.append((px, py))
            
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], 
                        self.colors['graph_confidence'], 2)
        
        return y + graph_height + 20
    
    def _draw_session_stats(self, frame, x, y):
        """Dibuja estadísticas de la sesión"""
        cv2.putText(frame, "ESTADISTICAS", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Duración de sesión
        session_duration = time.time() - self.session_start
        duration_str = f"{int(session_duration // 3600)}:{int((session_duration % 3600) // 60):02d}:{int(session_duration % 60):02d}"
        cv2.putText(frame, f"Sesion: {duration_str}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Tasa de reconocimiento
        if self.total_recognitions > 0:
            recognition_rate = (self.successful_recognitions / self.total_recognitions) * 100
            rate_color = self._get_rate_color(recognition_rate)
            cv2.putText(frame, f"Reconocidos: {recognition_rate:.0f}%", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       rate_color, 1)
        
        y += 20
        
        # Operadores vistos
        cv2.putText(frame, f"Operadores: {len(self.operators_seen)}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Detecciones desconocidas
        if self.unknown_detections > 0:
            cv2.putText(frame, f"Desconocidos: {self.unknown_detections}", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['warning'], 1)
        
        return y + 30
    
    def _update_statistics(self, result):
        """Actualiza las estadísticas internas"""
        self.total_recognitions += 1
        
        operator_info = result.get('operator_info')
        
        if operator_info:
            if operator_info.get('is_registered', False):
                # Operador registrado
                self.successful_recognitions += 1
                confidence = operator_info.get('confidence', 0)
                self.confidence_history.append(confidence)
                
                # Actualizar operador actual
                operator_id = operator_info['id']
                if self.current_operator_id != operator_id:
                    self.current_operator_id = operator_id
                    self.current_operator_start = time.time()
                    self.operators_seen.add(operator_id)
                
                # Reset contador de desconocido
                self.unknown_start_time = None
            else:
                # Operador NO registrado
                self.unknown_detections += 1
                self.confidence_history.append(0)
                
                # Iniciar contador de desconocido
                if self.unknown_start_time is None:
                    self.unknown_start_time = time.time()
                
                # Limpiar operador actual
                self.current_operator_id = None
                self.current_operator_start = None
        else:
            # Sin detección
            self.confidence_history.append(0)
            self.unknown_start_time = None
            self.current_operator_id = None
            self.current_operator_start = None
    
    def _get_confidence_color(self, confidence):
        """Obtiene color según nivel de confianza"""
        if confidence > 0.7:
            return self.colors['success']
        elif confidence > 0.5:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _get_rate_color(self, rate):
        """Obtiene color según tasa de reconocimiento"""
        if rate > 80:
            return self.colors['success']
        elif rate > 60:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def reset(self):
        """Reinicia el dashboard"""
        self.confidence_history.clear()
        self.session_start = time.time()
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.unknown_detections = 0
        self.operators_seen.clear()
        self.current_operator_start = None
        self.current_operator_id = None
        self.unknown_start_time = None