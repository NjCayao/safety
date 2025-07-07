"""
Master Dashboard
===============
Dashboard principal que integra todos los módulos de monitoreo.
Diseñado para mostrarse en el lado derecho de la pantalla.
"""

import cv2
import numpy as np
import time
from collections import deque

class MasterDashboard:
    def __init__(self, width=300, position='left', enable_analysis_dashboard=True):
        """
        Inicializa el dashboard maestro.
        
        Args:
            width: Ancho del panel del dashboard (default: 300px)
            position: Posición del dashboard ('left' o 'right', default: 'left')
            enable_analysis_dashboard: Si habilitar el dashboard de análisis
        """
        self.width = width
        self.position = position
        self.margin = 10
        
        # Inicializar AnalysisDashboard si está disponible
        self.analysis_dashboard = None
        if enable_analysis_dashboard:
            try:
                from core.analysis.analysis_dashboard import AnalysisDashboard
                self.analysis_dashboard = AnalysisDashboard(panel_width=300, position='right')
                print("AnalysisDashboard inicializado en el lado derecho")
            except ImportError:
                print("AnalysisDashboard no disponible - continuando sin él")
        
        # Colores del tema
        self.colors = {
            'background': (20, 20, 20),
            'panel_bg': (30, 30, 30),
            'section_bg': (25, 25, 25),
            'text_primary': (255, 255, 255),
            'text_secondary': (180, 180, 180),
            'success': (0, 255, 0),
            'warning': (0, 165, 255),
            'danger': (0, 0, 255),
            'accent': (255, 255, 0),
            'fatigue_color': (255, 150, 150),
            'behavior_color': (150, 150, 255),
            'graph_bg': (40, 40, 40),
            'graph_line': (0, 255, 255)
        }
        
        # Fuentes - Tamaño pequeño para texto compacto
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_sizes = {
            'title': 0.5,      # ~12px
            'section': 0.45,   # ~11px
            'body': 0.4,       # ~10px
            'small': 0.35      # ~9px
        }
        
        # Historiales para gráficos
        self.fatigue_history = deque(maxlen=60)
        self.behavior_history = deque(maxlen=60)
        self.alert_history = deque(maxlen=10)
        
        # Secciones del dashboard
        self.sections = {
            'header': {'height': 40},
            'operator': {'height': 50},
            'fatigue': {'height': 180},
            'behavior': {'height': 180},
            'statistics': {'height': 120},
            'alerts': {'height': 100}
        }
        
        # Calcular altura total necesaria
        self.height = sum(s['height'] for s in self.sections.values()) + (len(self.sections) + 1) * 5
        
    def render(self, frame, fatigue_result=None, behavior_result=None, operator_info=None, analysis_data=None):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame de video original
            fatigue_result: Resultado del análisis de fatiga
            behavior_result: Resultado del análisis de comportamientos
            operator_info: Información del operador actual
            analysis_data: Datos del análisis facial (para AnalysisDashboard)
            
        Returns:
            numpy.ndarray: Frame con dashboard integrado
        """
        # Primero renderizar el dashboard principal (izquierdo)
        h, w = frame.shape[:2]
        
        # Determinar posición X del dashboard
        if self.position == 'left':
            dashboard_x = self.margin
        else:
            dashboard_x = w - self.width - self.margin
        
        # Crear overlay para el dashboard
        overlay = frame.copy()
        
        # Fondo principal del dashboard
        cv2.rectangle(overlay, 
                     (dashboard_x, self.margin), 
                     (dashboard_x + self.width, h - self.margin),
                     self.colors['background'], -1)
        
        # Aplicar transparencia
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Actualizar historiales
        self._update_histories(fatigue_result, behavior_result)
        
        # Renderizar secciones del dashboard principal
        y_offset = self.margin + 5
        
        # 1. Header
        y_offset = self._draw_header(frame, dashboard_x, y_offset)
        
        # 2. Información del operador
        y_offset = self._draw_operator_section(frame, dashboard_x, y_offset, operator_info)
        
        # 3. Módulo de Fatiga
        y_offset = self._draw_fatigue_section(frame, dashboard_x, y_offset, fatigue_result)
        
        # 4. Módulo de Comportamientos
        y_offset = self._draw_behavior_section(frame, dashboard_x, y_offset, behavior_result)
        
        # 5. Estadísticas combinadas
        y_offset = self._draw_statistics_section(frame, dashboard_x, y_offset, fatigue_result, behavior_result)
        
        # 6. Alertas recientes
        y_offset = self._draw_alerts_section(frame, dashboard_x, y_offset)
        
        # Indicadores visuales en el video principal
        self._draw_video_overlays(frame, fatigue_result, behavior_result)
        
        # NUEVO: Renderizar AnalysisDashboard si está disponible
        if self.analysis_dashboard and analysis_data:
            frame = self.analysis_dashboard.render(frame, analysis_data)
        
        return frame
    
    def _draw_header(self, frame, x, y):
        """Dibuja el encabezado del dashboard"""
        # Fondo de sección
        section_height = self.sections['header']['height']
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título
        title = "SISTEMA DE MONITOREO"
        text_size = cv2.getTextSize(title, self.font, self.font_sizes['title'], 1)[0]
        text_x = x + (self.width - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, y + 25),
                   self.font, self.font_sizes['title'],
                   self.colors['accent'], 1)
        
        # Hora actual
        time_str = time.strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (x + self.width - 60, y + 25),
                   self.font, self.font_sizes['body'],
                   self.colors['text_secondary'], 1)
        
        return y + section_height + 5
    
    def _draw_operator_section(self, frame, x, y, operator_info):
        """Dibuja la sección de información del operador"""
        section_height = self.sections['operator']['height']
        
        # Título de sección
        cv2.putText(frame, "OPERADOR", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['text_secondary'], 1)
        
        # Línea separadora
        cv2.line(frame, (x + 10, y + 20), (x + self.width - 10, y + 20),
                self.colors['text_secondary'], 1)
        
        if operator_info:
            # Nombre
            name = operator_info.get('name', 'Desconocido').replace('ñ', 'n')
            cv2.putText(frame, f"Nombre: {name}", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['text_primary'], 1)
            
            # ID
            cv2.putText(frame, f"ID: {operator_info.get('id', 'N/A')}", (x + 15, y + 48),
                       self.font, self.font_sizes['small'],
                       self.colors['text_secondary'], 1)
        else:
            cv2.putText(frame, "No identificado", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['warning'], 1)
        
        return y + section_height + 5
    
    def _draw_fatigue_section(self, frame, x, y, result):
        """Dibuja la sección de fatiga"""
        section_height = self.sections['fatigue']['height']
        
        # Fondo de sección
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título con indicador de estado
        status_color = self.colors['success']
        if result and result.get('is_critical', False):
            status_color = self.colors['danger']
        elif result and result.get('microsleep_count', 0) > 0:
            status_color = self.colors['warning']
        
        cv2.putText(frame, "MODULO FATIGA", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['fatigue_color'], 1)
        
        # Indicador de estado
        cv2.circle(frame, (x + self.width - 20, y + 12), 4, status_color, -1)
        
        if not result:
            cv2.putText(frame, "Sin datos", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['text_secondary'], 1)
            return y + section_height + 5
        
        y_offset = y + 30
        
        # EAR
        ear_value = result.get('ear_value', 0)
        ear_threshold = result.get('ear_threshold', 0.25)
        ear_text = f"EAR: {ear_value:.3f} (Umbral: {ear_threshold:.3f})"
        ear_color = self.colors['danger'] if ear_value < ear_threshold else self.colors['success']
        cv2.putText(frame, ear_text, (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   ear_color, 1)
        
        # Barra visual de EAR
        y_offset += 15
        self._draw_progress_bar(frame, x + 15, y_offset, self.width - 30, 8,
                               ear_value, ear_threshold, ear_color, max_value=0.4)
        
        # Nivel de fatiga
        y_offset += 20
        fatigue_pct = result.get('fatigue_percentage', 0)
        fatigue_color = self._get_level_color(fatigue_pct, [40, 60, 80])
        cv2.putText(frame, f"Nivel Fatiga: {fatigue_pct}%", (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   fatigue_color, 1)
        
        # Barra de fatiga
        y_offset += 15
        self._draw_progress_bar(frame, x + 15, y_offset, self.width - 30, 8,
                               fatigue_pct, 60, fatigue_color, max_value=100)
        
        # Microsueños
        y_offset += 20
        microsleeps = result.get('microsleep_count', 0)
        ms_color = self._get_level_color(microsleeps, [1, 2, 3])
        cv2.putText(frame, f"Microsuenos (10min): {microsleeps}/3", (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   ms_color, 1)
        
        # Estado de ojos
        if result.get('eyes_closed', False):
            y_offset += 15
            duration = result.get('eyes_closed_duration', 0)
            cv2.putText(frame, f"Ojos cerrados: {duration:.1f}s", (x + 15, y_offset),
                       self.font, self.font_sizes['small'],
                       self.colors['danger'], 1)
        
        # Mini gráfico de historial
        y_offset += 20
        self._draw_mini_graph(frame, x + 15, y_offset, self.width - 30, 40,
                            self.fatigue_history, "Historial Fatiga (60s)",
                            self.colors['fatigue_color'])
        
        return y + section_height + 5
    
    def _draw_behavior_section(self, frame, x, y, result):
        """Dibuja la sección de comportamientos"""
        section_height = self.sections['behavior']['height']
        
        # Fondo de sección
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título con indicador
        has_alerts = result and len(result.get('alerts', [])) > 0
        status_color = self.colors['danger'] if has_alerts else self.colors['success']
        
        cv2.putText(frame, "MODULO COMPORTAMIENTOS", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['behavior_color'], 1)
        
        cv2.circle(frame, (x + self.width - 20, y + 12), 4, status_color, -1)
        
        if not result:
            cv2.putText(frame, "Sin datos", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['text_secondary'], 1)
            return y + section_height + 5
        
        y_offset = y + 30
        
        # Detecciones actuales
        detections = result.get('detections', [])
        phone_detected = any(d[0] == 'cell phone' for d in detections)
        cigarette_detected = any(d[0] == 'cigarette' for d in detections)
        
        # Teléfono
        phone_color = self.colors['danger'] if phone_detected else self.colors['text_secondary']
        phone_status = "DETECTADO" if phone_detected else "No detectado"
        cv2.putText(frame, f"Telefono: {phone_status}", (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   phone_color, 1)
        
        # Duración teléfono
        detector_info = result.get('detector_info', {})
        durations = detector_info.get('behavior_durations', {})
        if 'cell phone' in durations:
            y_offset += 15
            duration = durations['cell phone']
            duration_color = self._get_level_color(duration, [3, 5, 7])
            cv2.putText(frame, f"  Duracion: {duration:.1f}s", (x + 25, y_offset),
                       self.font, self.font_sizes['small'],
                       duration_color, 1)
            
            # Barra de progreso
            y_offset += 10
            self._draw_progress_bar(frame, x + 25, y_offset, self.width - 40, 6,
                                   duration, 7, duration_color, max_value=10)
        
        # Cigarrillo
        y_offset += 20
        cig_color = self.colors['warning'] if cigarette_detected else self.colors['text_secondary']
        cig_status = "DETECTADO" if cigarette_detected else "No detectado"
        cv2.putText(frame, f"Cigarrillo: {cig_status}", (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   cig_color, 1)
        
        # Frecuencia cigarrillo
        if detector_info.get('cigarette_detections', 0) > 0:
            y_offset += 15
            detections_count = detector_info.get('cigarette_detections', 0)
            threshold = detector_info.get('cigarette_pattern_threshold', 3)
            cv2.putText(frame, f"  Detecciones: {detections_count}/{threshold}", (x + 25, y_offset),
                       self.font, self.font_sizes['small'],
                       self.colors['text_secondary'], 1)
        
        # Mini gráfico
        y_offset += 25
        self._draw_mini_graph(frame, x + 15, y_offset, self.width - 30, 40,
                            self.behavior_history, "Actividad (60s)",
                            self.colors['behavior_color'])
        
        return y + section_height + 5
    
    def _draw_statistics_section(self, frame, x, y, fatigue_result, behavior_result):
        """Dibuja estadísticas combinadas"""
        section_height = self.sections['statistics']['height']
        
        # Título
        cv2.putText(frame, "ESTADISTICAS", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['text_secondary'], 1)
        
        cv2.line(frame, (x + 10, y + 20), (x + self.width - 10, y + 20),
                self.colors['text_secondary'], 1)
        
        y_offset = y + 35
        
        # Estado general
        status_parts = []
        if fatigue_result and fatigue_result.get('is_critical', False):
            status_parts.append("FATIGA CRITICA")
        if behavior_result and behavior_result.get('alerts'):
            status_parts.append("ALERTA COMPORTAMIENTO")
        
        if not status_parts:
            status_text = "Estado: NORMAL"
            status_color = self.colors['success']
        else:
            status_text = "Estado: " + ", ".join(status_parts)
            status_color = self.colors['danger']
        
        cv2.putText(frame, status_text, (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   status_color, 1)
        
        # Modo día/noche
        y_offset += 15
        is_night = False
        light_level = 0
        if fatigue_result:
            is_night = fatigue_result.get('is_night_mode', False)
            light_level = fatigue_result.get('light_level', 0)
        elif behavior_result:
            is_night = behavior_result.get('is_night_mode', False)
            light_level = behavior_result.get('light_level', 0)
        
        mode_text = f"Modo: {'NOCTURNO' if is_night else 'DIURNO'} (Luz: {light_level:.0f})"
        mode_color = (100, 100, 255) if is_night else (255, 200, 0)
        cv2.putText(frame, mode_text, (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   mode_color, 1)
        
        # Calibración
        y_offset += 15
        fatigue_calib = fatigue_result and fatigue_result.get('is_calibrated', False)
        behavior_calib = behavior_result and behavior_result.get('is_calibrated', False)
        
        if fatigue_calib and behavior_calib:
            calib_text = "Calibracion: COMPLETA"
            calib_color = self.colors['success']
        elif fatigue_calib or behavior_calib:
            calib_text = "Calibracion: PARCIAL"
            calib_color = self.colors['warning']
        else:
            calib_text = "Calibracion: PENDIENTE"
            calib_color = self.colors['danger']
        
        cv2.putText(frame, calib_text, (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   calib_color, 1)
        
        # Tiempo de sesión
        y_offset += 15
        session_time = time.strftime("%M:%S", time.gmtime(time.time() % 3600))
        cv2.putText(frame, f"Tiempo sesion: {session_time}", (x + 15, y_offset),
                   self.font, self.font_sizes['small'],
                   self.colors['text_secondary'], 1)
        
        return y + section_height + 5
    
    def _draw_alerts_section(self, frame, x, y):
        """Dibuja las alertas recientes"""
        section_height = self.sections['alerts']['height']
        
        # Fondo de sección
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título
        alert_count = len(self.alert_history)
        title_color = self.colors['danger'] if alert_count > 0 else self.colors['text_secondary']
        cv2.putText(frame, f"ALERTAS RECIENTES ({alert_count})", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   title_color, 1)
        
        if not self.alert_history:
            cv2.putText(frame, "Sin alertas", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['success'], 1)
            return y + section_height + 5
        
        # Mostrar últimas 4 alertas
        y_offset = y + 30
        for i, alert in enumerate(list(self.alert_history)[-4:]):
            alert_type = alert['type']
            alert_msg = alert['message'].replace('ñ', 'n')
            time_ago = int(time.time() - alert['timestamp'])
            
            # Color según tipo
            color = self.colors['fatigue_color'] if alert_type == 'fatigue' else self.colors['behavior_color']
            
            # Texto compacto
            text = f"[{time_ago}s] {alert_msg[:30]}"
            cv2.putText(frame, text, (x + 15, y_offset),
                       self.font, self.font_sizes['small'],
                       color, 1)
            y_offset += 15
        
        return y + section_height + 5
    
    def _draw_video_overlays(self, frame, fatigue_result, behavior_result):
        """Dibuja overlays en el video principal"""
        # Alertas críticas
        if fatigue_result and fatigue_result.get('is_critical', False):
            # Marco rojo sutil
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (2, 2), (w - 2, h - 2),
                         self.colors['danger'], 2)
            
            # Mensaje de alerta
            msg = "FATIGA CRITICA DETECTADA"
            text_size = cv2.getTextSize(msg, self.font, self.font_sizes['title'], 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, 30),
                       self.font, self.font_sizes['title'],
                       self.colors['danger'], 2)
        
        # Alertas de comportamiento
        if behavior_result and behavior_result.get('alerts'):
            for alert in behavior_result['alerts']:
                if '7s' in alert[0]:
                    msg = "ALERTA: USO PROLONGADO TELEFONO"
                    cv2.putText(frame, msg, (10, 60),
                               self.font, self.font_sizes['body'],
                               self.colors['danger'], 1)
    
    def _draw_progress_bar(self, frame, x, y, width, height, value, threshold, color, max_value=10):
        """Dibuja una barra de progreso compacta"""
        # Fondo
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['graph_bg'], -1)
        
        # Valor
        normalized = min(1.0, value / max_value)
        value_width = int(width * normalized)
        cv2.rectangle(frame, (x, y), (x + value_width, y + height),
                     color, -1)
        
        # Línea de umbral
        if threshold < max_value:
            threshold_x = x + int(width * (threshold / max_value))
            cv2.line(frame, (threshold_x, y - 2), (threshold_x, y + height + 2),
                    self.colors['warning'], 1)
    
    def _draw_mini_graph(self, frame, x, y, width, height, data, title, color):
        """Dibuja un gráfico miniatura"""
        # Título
        cv2.putText(frame, title, (x, y - 3),
                   self.font, self.font_sizes['small'],
                   self.colors['text_secondary'], 1)
        
        # Fondo del gráfico
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['graph_bg'], -1)
        
        if len(data) < 2:
            return
        
        # Dibujar línea
        points = []
        for i, value in enumerate(data):
            px = x + int(i * width / len(data))
            py = y + height - int(value * height)
            points.append((px, py))
        
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], color, 1)
    
    def _get_level_color(self, value, thresholds):
        """Obtiene color según nivel"""
        if value >= thresholds[2]:
            return self.colors['danger']
        elif value >= thresholds[1]:
            return self.colors['warning']
        elif value >= thresholds[0]:
            return self.colors['text_primary']
        else:
            return self.colors['success']
    
    def _update_histories(self, fatigue_result, behavior_result):
        """Actualiza los historiales"""
        # Fatiga
        if fatigue_result:
            self.fatigue_history.append(fatigue_result.get('fatigue_percentage', 0) / 100)
            
            if fatigue_result.get('microsleep_detected', False):
                self.alert_history.append({
                    'type': 'fatigue',
                    'message': 'Microsueno detectado',
                    'timestamp': time.time()
                })
        
        # Comportamiento
        if behavior_result:
            has_detection = len(behavior_result.get('detections', [])) > 0
            self.behavior_history.append(1.0 if has_detection else 0.0)
            
            for alert in behavior_result.get('alerts', []):
                alert_msg = {
                    'phone_3s': 'Telefono 3s',
                    'phone_7s': 'Telefono 7s critico',
                    'smoking_pattern': 'Patron cigarrillo',
                    'smoking_7s': 'Cigarrillo continuo'
                }.get(alert[0], alert[0])
                
                self.alert_history.append({
                    'type': 'behavior',
                    'message': alert_msg,
                    'timestamp': time.time()
                })