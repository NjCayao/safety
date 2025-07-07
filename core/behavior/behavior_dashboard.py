"""
Dashboard de Comportamientos
============================
Visualización especializada para el módulo de comportamientos.
"""

import cv2
import numpy as np
from collections import deque
import time

class BehaviorDashboard:
    def __init__(self, width=350, position='left'):
        """
        Inicializa el dashboard de comportamientos.
        
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
            'phone_color': (0, 0, 255),      # Rojo para teléfono
            'cigarette_color': (0, 165, 255), # Naranja para cigarrillo
            'graph_bg': (40, 40, 40),
            'graph_phone': (0, 100, 255),
            'graph_cigarette': (0, 200, 255)
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
        
        # Historial para gráficos (últimos 60 segundos)
        self.phone_history = deque(maxlen=60)
        self.cigarette_history = deque(maxlen=60)
        self.alert_history = deque(maxlen=10)
        
        # Contadores para estadísticas
        self.session_start = time.time()
        self.total_phone_detections = 0
        self.total_cigarette_detections = 0
        self.total_alerts = 0
        
        # Cache para elementos estáticos
        self.static_cache = None
        self.last_cache_update = 0
        
    def render(self, frame, analysis_result):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame donde dibujar
            analysis_result: Resultado del análisis de comportamientos
            
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
        y_offset = self._draw_title(frame, panel_x, y_offset)
        
        # Estado del operador
        y_offset = self._draw_operator_status(frame, panel_x, y_offset, analysis_result)
        
        # Detecciones actuales
        y_offset = self._draw_current_detections(frame, panel_x, y_offset, analysis_result)
        
        # Contadores de tiempo
        y_offset = self._draw_behavior_timers(frame, panel_x, y_offset, analysis_result)
        
        # Gráfico de actividad
        y_offset = self._draw_activity_graph(frame, panel_x, y_offset)
        
        # Estadísticas de sesión
        y_offset = self._draw_session_stats(frame, panel_x, y_offset)
        
        # Alertas recientes
        y_offset = self._draw_recent_alerts(frame, panel_x, y_offset)
        
        # Indicadores en el video principal
        self._draw_video_overlays(frame, analysis_result)
        
        return frame
    
    def _draw_title(self, frame, x, y):
        """Dibuja el título del panel"""
        title = "MONITOR COMPORTAMIENTOS"
        
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
        
        # Estado de calibración y modo
        is_calibrated = result.get('is_calibrated', False)
        is_night = result.get('is_night_mode', False)
        
        # Calibración
        calib_text = "Calibrado" if is_calibrated else "Sin calibrar"
        calib_color = self.colors['success'] if is_calibrated else self.colors['warning']
        cv2.putText(frame, f"Estado: {calib_text}", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   calib_color, 1)
        
        # Modo día/noche
        mode_text = "Modo: NOCHE" if is_night else "Modo: DÍA"
        mode_color = (100, 100, 255) if is_night else (255, 200, 0)
        cv2.putText(frame, mode_text, (x + 150, y),
                   self.fonts['small'], self.font_scales['small'],
                   mode_color, 1)
        
        return y + 30
    
    def _draw_current_detections(self, frame, x, y, result):
        """Dibuja las detecciones actuales"""
        cv2.putText(frame, "DETECCIONES ACTUALES", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        detections = result.get('detections', [])
        
        # Contar detecciones por tipo
        phone_count = sum(1 for d in detections if d[0] == 'cell phone')
        cigarette_count = sum(1 for d in detections if d[0] == 'cigarette')
        
        # Mostrar teléfono
        phone_color = self.colors['phone_color'] if phone_count > 0 else self.colors['text_secondary']
        cv2.putText(frame, f"Telefono: {'DETECTADO' if phone_count > 0 else 'No detectado'}", 
                   (x + 30, y), self.fonts['body'], self.font_scales['body'],
                   phone_color, 1 if phone_count == 0 else 2)
        
        y += 25
        
        # Mostrar cigarrillo
        cig_color = self.colors['cigarette_color'] if cigarette_count > 0 else self.colors['text_secondary']
        cv2.putText(frame, f"Cigarrillo: {'DETECTADO' if cigarette_count > 0 else 'No detectado'}", 
                   (x + 30, y), self.fonts['body'], self.font_scales['body'],
                   cig_color, 1 if cigarette_count == 0 else 2)
        
        return y + 30
    
    def _draw_behavior_timers(self, frame, x, y, result):
        """Dibuja los contadores de tiempo para comportamientos"""
        cv2.putText(frame, "DURACION COMPORTAMIENTOS", (x + 20, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_secondary'], 1)
        
        y += 20
        
        # Obtener información del detector
        detector_info = result.get('detector_info', {})
        durations = detector_info.get('behavior_durations', {})
        
        # Timer de teléfono
        if 'cell phone' in durations:
            duration = durations['cell phone']
            
            # Barra de progreso para teléfono
            self._draw_progress_bar(frame, x + 30, y, 
                                  "Telefono", duration, 
                                  [3, 7],  # Umbrales
                                  self.colors['phone_color'])
            y += 35
        
        # Información de cigarrillo
        if 'cigarette' in durations or 'cigarette_detections' in detector_info:
            # Frecuencia de detección
            cig_detections = detector_info.get('cigarette_detections', 0)
            cig_threshold = detector_info.get('cigarette_pattern_threshold', 3)
            
            # Barra de frecuencia
            self._draw_frequency_bar(frame, x + 30, y,
                                   "Cigarrillo", cig_detections, cig_threshold,
                                   self.colors['cigarette_color'])
            y += 35
        
        return y + 10
    
    def _draw_progress_bar(self, frame, x, y, label, value, thresholds, color):
        """Dibuja una barra de progreso con umbrales"""
        bar_width = self.width - 80
        bar_height = 15
        
        # Etiqueta y valor
        cv2.putText(frame, f"{label}: {value:.1f}s", (x, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 5
        
        # Fondo de la barra
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['graph_bg'], -1)
        
        # Valor actual (normalizado al máximo umbral)
        max_threshold = max(thresholds) if thresholds else 10
        normalized_value = min(1.0, value / max_threshold)
        value_width = int(bar_width * normalized_value)
        
        # Color según umbrales
        if value >= thresholds[-1]:
            bar_color = self.colors['danger']
        elif value >= thresholds[0]:
            bar_color = self.colors['warning']
        else:
            bar_color = color
        
        cv2.rectangle(frame, (x, y), (x + value_width, y + bar_height),
                     bar_color, -1)
        
        # Líneas de umbral
        for threshold in thresholds:
            threshold_x = x + int(bar_width * (threshold / max_threshold))
            cv2.line(frame, (threshold_x, y - 2), (threshold_x, y + bar_height + 2),
                    self.colors['text_secondary'], 1)
    
    def _draw_frequency_bar(self, frame, x, y, label, count, threshold, color):
        """Dibuja una barra de frecuencia"""
        bar_width = self.width - 80
        bar_height = 15
        
        # Etiqueta
        cv2.putText(frame, f"{label}: {count}/{threshold} detecciones", (x, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 5
        
        # Fondo
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['graph_bg'], -1)
        
        # Valor
        normalized = min(1.0, count / threshold)
        value_width = int(bar_width * normalized)
        bar_color = self.colors['danger'] if count >= threshold else color
        
        cv2.rectangle(frame, (x, y), (x + value_width, y + bar_height),
                     bar_color, -1)
    
    def _draw_activity_graph(self, frame, x, y):
        """Dibuja gráfico de actividad temporal"""
        if len(self.phone_history) < 2:
            return y
        
        cv2.putText(frame, "ACTIVIDAD (60s)", (x + 20, y),
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
        
        # Dibujar datos de teléfono
        if self.phone_history:
            points_phone = []
            for i, value in enumerate(self.phone_history):
                px = graph_x + int(i * graph_width / len(self.phone_history))
                py = graph_y + graph_height - int(value * graph_height)
                points_phone.append((px, py))
            
            # Línea de teléfono
            for i in range(1, len(points_phone)):
                cv2.line(frame, points_phone[i-1], points_phone[i], 
                        self.colors['graph_phone'], 2)
        
        # Dibujar datos de cigarrillo
        if self.cigarette_history:
            points_cig = []
            for i, value in enumerate(self.cigarette_history):
                px = graph_x + int(i * graph_width / len(self.cigarette_history))
                py = graph_y + graph_height - int(value * graph_height * 0.5)  # Escala diferente
                points_cig.append((px, py))
            
            # Línea de cigarrillo
            for i in range(1, len(points_cig)):
                cv2.line(frame, points_cig[i-1], points_cig[i], 
                        self.colors['graph_cigarette'], 1)
        
        # Leyenda
        y = graph_y + graph_height + 15
        cv2.putText(frame, "Tel", (graph_x, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['graph_phone'], 1)
        cv2.putText(frame, "Cig", (graph_x + 40, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['graph_cigarette'], 1)
        
        return y + 20
    
    def _draw_session_stats(self, frame, x, y):
        """Dibuja estadísticas de la sesión"""
        cv2.putText(frame, "ESTADISTICAS SESION", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['text_secondary'], 1)
        
        y += 25
        
        # Duración de sesión
        session_duration = time.time() - self.session_start
        duration_str = f"{int(session_duration // 60)}:{int(session_duration % 60):02d}"
        cv2.putText(frame, f"Duracion: {duration_str}", (x + 30, y),
                   self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Total detecciones
        cv2.putText(frame, f"Detecciones telefono: {self.total_phone_detections}", 
                   (x + 30, y), self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        cv2.putText(frame, f"Detecciones cigarrillo: {self.total_cigarette_detections}", 
                   (x + 30, y), self.fonts['small'], self.font_scales['small'],
                   self.colors['text_primary'], 1)
        
        y += 20
        
        # Total alertas
        alert_color = self.colors['danger'] if self.total_alerts > 5 else self.colors['warning'] if self.total_alerts > 2 else self.colors['text_primary']
        cv2.putText(frame, f"Total alertas: {self.total_alerts}", 
                   (x + 30, y), self.fonts['small'], self.font_scales['small'],
                   alert_color, 1)
        
        return y + 30
    
    def _draw_recent_alerts(self, frame, x, y):
        """Dibuja las alertas recientes"""
        if not self.alert_history:
            return y
        
        cv2.putText(frame, "ALERTAS RECIENTES", (x + 20, y),
                   self.fonts['body'], self.font_scales['body'],
                   self.colors['danger'], 1)
        
        y += 25
        
        # Mostrar últimas 3 alertas
        for alert in list(self.alert_history)[-3:]:
            alert_type = alert['type']
            alert_time = alert['time']
            time_ago = int(time.time() - alert_time)
            
            # Icono según tipo
            if 'phone' in alert_type:
                icon = "TEL"
                color = self.colors['phone_color']
            elif 'smoking' in alert_type:
                icon = "CIG"
                color = self.colors['cigarette_color']
            else:
                icon = "!"
                color = self.colors['warning']
            
            # Texto de alerta
            cv2.putText(frame, f"[{icon}]", (x + 30, y),
                       self.fonts['small'], self.font_scales['small'],
                       color, 2)
            
            alert_text = f"{alert.get('message', alert_type)} ({time_ago}s)"
            cv2.putText(frame, alert_text, (x + 60, y),
                       self.fonts['small'], self.font_scales['small'],
                       self.colors['text_primary'], 1)
            
            y += 20
        
        return y + 10
    
    def _draw_video_overlays(self, frame, result):
        """Dibuja overlays en el video principal"""
        alerts = result.get('alerts', [])
        
        # Si hay alertas activas, agregar indicadores
        if alerts:
            h, w = frame.shape[:2]
            
            for alert in alerts:
                alert_type = alert[0]
                
                # Borde de alerta según severidad
                if '7s' in alert_type or 'critical' in alert_type:
                    # Borde rojo para alertas críticas
                    thickness = 5
                    cv2.rectangle(frame, (thickness, thickness), 
                                 (w - thickness, h - thickness),
                                 self.colors['danger'], thickness)
                    
                    # Mensaje de alerta
                    if 'phone' in alert_type:
                        msg = "¡ALERTA! USO PROLONGADO DE TELEFONO"
                    else:
                        msg = "¡ALERTA! COMPORTAMIENTO DETECTADO"
                    
                    text_size = cv2.getTextSize(msg, self.fonts['title'], 
                                               self.font_scales['title'], 2)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = 50
                    
                    # Fondo para el texto
                    cv2.rectangle(frame, (text_x - 10, text_y - 30), 
                                 (text_x + text_size[0] + 10, text_y + 10),
                                 self.colors['danger'], -1)
                    
                    cv2.putText(frame, msg, (text_x, text_y),
                               self.fonts['title'], self.font_scales['title'],
                               self.colors['text_primary'], 2)
    
    def _update_histories(self, result):
        """Actualiza los historiales"""
        detections = result.get('detections', [])
        
        # Contar detecciones actuales
        phone_detected = any(d[0] == 'cell phone' for d in detections)
        cigarette_detected = any(d[0] == 'cigarette' for d in detections)
        
        # Actualizar historiales
        self.phone_history.append(1.0 if phone_detected else 0.0)
        self.cigarette_history.append(1.0 if cigarette_detected else 0.0)
        
        # Actualizar contadores totales
        if phone_detected:
            self.total_phone_detections += 1
        if cigarette_detected:
            self.total_cigarette_detections += 1
        
        # Procesar alertas
        alerts = result.get('alerts', [])
        for alert in alerts:
            self.alert_history.append({
                'type': alert[0],
                'message': self._get_alert_message(alert),
                'time': time.time()
            })
            self.total_alerts += 1
    
    def _get_alert_message(self, alert):
        """Obtiene mensaje legible para una alerta"""
        alert_type, behavior, value = alert
        
        messages = {
            'phone_3s': f"Teléfono {value:.1f}s",
            'phone_7s': f"Teléfono crítico {value:.1f}s",
            'smoking_pattern': f"Patrón cigarrillo x{value}",
            'smoking_7s': f"Cigarrillo continuo {value:.1f}s"
        }
        
        return messages.get(alert_type, alert_type)
    
    def reset(self):
        """Reinicia el dashboard"""
        self.phone_history.clear()
        self.cigarette_history.clear()
        self.alert_history.clear()
        self.session_start = time.time()
        self.total_phone_detections = 0
        self.total_cigarette_detections = 0
        self.total_alerts = 0
        self.static_cache = None