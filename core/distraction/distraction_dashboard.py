"""
Dashboard de Detección de Distracciones
==================================
Visualización especializada para el módulo de distracciones.
"""

import cv2
import numpy as np
from collections import deque
import time

class DistractionDashboard:
    def __init__(self, width=350, position='right'):
        """
        Inicializa el dashboard de distracciones.
        
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
            'center': (0, 255, 0),
            'distracted': (0, 0, 255),
            'extreme': (255, 0, 255),
            'graph_bg': (40, 40, 40),
            'graph_line': (0, 165, 255)
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
        self.direction_history = deque(maxlen=100)
        self.distraction_events = deque(maxlen=10)
        
        # Estadísticas de sesión
        self.session_start = time.time()
        self.total_distractions = 0
        self.time_distracted = 0
        self.last_distraction_start = None
        self.level2_events = 0
    
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
        
        # Gráfico de dirección
        y_offset = self._draw_direction_graph(frame, panel_x, y_offset)
        
        # Historial de distracciones
        y_offset = self._draw_distraction_history(frame, panel_x, y_offset, analysis_result)
        
        # Estadísticas de sesión
        y_offset = self._draw_session_stats(frame, panel_x, y_offset)
        
        # Alertas
        y_offset = self._draw_alerts(frame, panel_x, y_offset, analysis_result)
        
        return frame
    
    def _draw_title(self, frame, x, y):
        """Dibuja el título del panel"""
        title = "DETECCIÓN DE DISTRACCIONES"
        
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
        detector_status = result.get('detector_status', {})
        
        cv2.putText(frame, "MÉTRICAS", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Dirección actual
        direction = detector_status.get('direction', 'CENTRO')
        if direction == 'CENTRO':
            dir_color = self.colors['center']
        elif direction == 'EXTREMO':
            dir_color = self.colors['extreme']
        else:
            dir_color = self.colors['distracted']
        
        cv2.putText(frame, f"Dirección: {direction}", (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   dir_color, 1)
        
        y += 25
        
        # Nivel de alerta
        alert_level = detector_status.get('current_alert_level', 0)
        if alert_level == 0:
            level_text = "Normal"
            level_color = self.colors['success']
        elif alert_level == 1:
            level_text = "Nivel 1"
            level_color = self.colors['warning']
        else:
            level_text = "Nivel 2"
            level_color = self.colors['danger']
        
        cv2.putText(frame, f"Alerta: {level_text}", (x + 30, y),
                   self.fonts['body'], self.font_scales['body'],
                   level_color, 1)
        
        # Tiempo de distracción actual
        y += 25
        distraction_time = detector_status.get('distraction_time', 0)
        if distraction_time > 0:
            cv2.putText(frame, f"Tiempo: {distraction_time:.1f}s", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['warning'], 1)
            
            # Barra de progreso
            y += 5
            bar_width = self.width - 60
            bar_height = 10
            
            # Obtener tiempos de configuración
            level1_time = detector_status.get('level1_time', 3)
            level2_time = detector_status.get('level2_time', 7)
            
            # Calcular progreso
            if distraction_time <= level1_time:
                # Progreso hacia nivel 1
                progress = distraction_time / level1_time
                bar_color = self.colors['warning']
                progress_text = f"Hacia Nivel 1: {distraction_time:.1f}/{level1_time}s"
            else:
                # Progreso hacia nivel 2
                progress = distraction_time / level2_time
                bar_color = self.colors['danger']
                progress_text = f"Hacia Nivel 2: {distraction_time:.1f}/{level2_time}s"
            
            # Fondo de la barra
            cv2.rectangle(frame, (x + 30, y), (x + 30 + bar_width, y + bar_height),
                         self.colors['graph_bg'], -1)
            
            # Progreso actual
            progress_width = int(bar_width * min(1.0, progress))
            cv2.rectangle(frame, (x + 30, y), (x + 30 + progress_width, y + bar_height),
                         bar_color, -1)
            
            # Marcador de nivel 1
            level1_x = x + 30 + int(bar_width * (level1_time / level2_time))
            cv2.line(frame, (level1_x, y - 2), (level1_x, y + bar_height + 2),
                    self.colors['warning'], 2)
            
            # Texto de progreso
            y += bar_height + 5
            cv2.putText(frame, progress_text, (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       bar_color, 1)
            
            y += 5
        
        # Confianza
        y += 15
        confidence = detector_status.get('confidence', 1.0)
        conf_color = self.colors['success'] if confidence > 0.8 else self.colors['warning']
        cv2.putText(frame, f"Confianza: {confidence:.2f}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   conf_color, 1)
        
        return y + 15
    
    def _draw_direction_graph(self, frame, x, y):
        """Dibuja gráfico histórico de dirección"""
        if len(self.direction_history) < 2:
            return y
        
        cv2.putText(frame, "HISTORIAL DIRECCIÓN", (x + 20, y),
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
        for i, (timestamp, direction) in enumerate(self.direction_history):
            px = graph_x + int(i * graph_width / len(self.direction_history))
            
            # Mapear dirección a altura
            if direction == 'CENTRO':
                py = graph_y + graph_height // 2
                color = self.colors['center']
            elif direction == 'IZQUIERDA':
                py = graph_y + int(graph_height * 0.25)
                color = self.colors['distracted']
            elif direction == 'DERECHA':
                py = graph_y + int(graph_height * 0.75)
                color = self.colors['distracted']
            else:  # EXTREMO
                py = graph_y + 5
                color = self.colors['extreme']
            
            cv2.circle(frame, (px, py), 2, color, -1)
        
        # Línea central
        cv2.line(frame, (graph_x, graph_y + graph_height // 2), 
                (graph_x + graph_width, graph_y + graph_height // 2),
                self.colors['text_secondary'], 1)
        
        return y + graph_height + 20
    
    def _draw_distraction_history(self, frame, x, y, result):
        """Dibuja historial de distracciones"""
        total_distractions = result.get('total_distractions', 0)
        window_minutes = result.get('window_minutes', 10)
        
        cv2.putText(frame, f"DISTRACCIONES ({window_minutes} min)", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Contador
        count_color = self._get_count_color(total_distractions)
        cv2.putText(frame, f"{total_distractions} / 3", (x + 30, y),
                   self.fonts['title'], self.font_scales['title'],
                   count_color, 2)
        
        y += 30
        
        # Últimas distracciones
        if self.distraction_events:
            cv2.putText(frame, "Últimas:", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_secondary'], 1)
            y += 20
            
            current_time = time.time()
            for i, event in enumerate(list(self.distraction_events)[-3:]):
                time_ago = current_time - event['timestamp']
                minutes = int(time_ago / 60)
                seconds = int(time_ago % 60)
                
                level_str = f"Nivel {event['level']}"
                cv2.putText(frame, f"  {level_str} - Hace {minutes}:{seconds:02d}", (x + 40, y),
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
        
        # Total de distracciones
        cv2.putText(frame, f"Total distracciones: {self.total_distractions}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Tiempo distraído
        if session_duration > 0:
            distraction_pct = (self.time_distracted / session_duration) * 100
            cv2.putText(frame, f"Tiempo distraído: {distraction_pct:.1f}%", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_primary'], 1)
        
        y += 20
        
        # Eventos nivel 2
        cv2.putText(frame, f"Alertas nivel 2: {self.level2_events}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        return y + 25
    
    def _draw_alerts(self, frame, x, y, result):
        """Dibuja alertas activas"""
        if result.get('multiple_distractions', False):
            # Alerta de múltiples distracciones
            cv2.putText(frame, "⚠ ALERTA DISTRACCIÓN", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['danger'], 2)
            
            y += 25
            cv2.putText(frame, "Múltiples distracciones detectadas", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['danger'], 1)
            
            y += 20
            cv2.putText(frame, "Recomendación: Mantener atención", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['warning'], 1)
        
        elif result.get('detector_status', {}).get('current_alert_level', 0) >= 2:
            # Alerta nivel 2
            cv2.putText(frame, "⚠ DISTRACCIÓN PROLONGADA", (x + 20, y),
                       self.fonts['body'], self.font_scales['body'],
                       self.colors['danger'], 2)
        
        return y + 30
    
    def _update_histories(self, result):
        """Actualiza los historiales"""
        detector_status = result.get('detector_status', {})
        
        # Actualizar historial de dirección
        direction = detector_status.get('direction', 'CENTRO')
        self.direction_history.append((time.time(), direction))
        
        # Actualizar tiempo distraído
        is_distracted = detector_status.get('is_distracted', False)
        if is_distracted:
            if self.last_distraction_start is None:
                self.last_distraction_start = time.time()
        else:
            if self.last_distraction_start is not None:
                self.time_distracted += time.time() - self.last_distraction_start
                self.last_distraction_start = None
        
        # Registrar eventos de distracción
        alert_level = detector_status.get('current_alert_level', 0)
        if alert_level > 0 and not any(e['active'] for e in self.distraction_events):
            # Nueva distracción
            self.distraction_events.append({
                'timestamp': time.time(),
                'level': alert_level,
                'active': True
            })
            self.total_distractions += 1
            
            if alert_level >= 2:
                self.level2_events += 1
        elif alert_level == 0:
            # Marcar distracciones como inactivas
            for event in self.distraction_events:
                event['active'] = False
    
    def _get_count_color(self, count):
        """Obtiene color según contador"""
        if count >= 3:
            return self.colors['danger']
        elif count >= 2:
            return self.colors['warning']
        else:
            return self.colors['success']
    
    def reset(self):
        """Reinicia el dashboard"""
        self.direction_history.clear()
        self.distraction_events.clear()
        self.session_start = time.time()
        self.total_distractions = 0
        self.time_distracted = 0
        self.last_distraction_start = None
        self.level2_events = 0