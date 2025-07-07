"""
Master Dashboard con Reconocimiento Facial
=========================================
Dashboard principal que integra todos los módulos incluyendo reconocimiento facial.
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
            'face_color': (150, 255, 150),  # Color para reconocimiento facial
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
        self.face_confidence_history = deque(maxlen=60)  # NUEVO
        self.alert_history = deque(maxlen=10)
        
        # Secciones del dashboard (ACTUALIZADO)
        self.sections = {
            'header': {'height': 40},
            'operator': {'height': 80},      # Aumentado para incluir info de reconocimiento
            'face_recognition': {'height': 100},  # NUEVA SECCIÓN
            'fatigue': {'height': 150},      # Reducido
            'behavior': {'height': 150},     # Reducido
            'statistics': {'height': 100},   # Reducido
            'alerts': {'height': 80}         # Reducido
        }
        
        # Calcular altura total necesaria
        self.height = sum(s['height'] for s in self.sections.values()) + (len(self.sections) + 1) * 5
        
    def render(self, frame, fatigue_result=None, behavior_result=None, face_result=None, analysis_data=None):
        """
        Renderiza el dashboard en el frame.
        
        Args:
            frame: Frame de video original
            fatigue_result: Resultado del análisis de fatiga
            behavior_result: Resultado del análisis de comportamientos
            face_result: Resultado del reconocimiento facial (NUEVO)
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
        self._update_histories(fatigue_result, behavior_result, face_result)
        
        # Renderizar secciones del dashboard principal
        y_offset = self.margin + 5
        
        # 1. Header
        y_offset = self._draw_header(frame, dashboard_x, y_offset)
        
        # 2. Información del operador (actualizada con reconocimiento facial)
        y_offset = self._draw_operator_section(frame, dashboard_x, y_offset, face_result)
        
        # 3. Módulo de Reconocimiento Facial (NUEVO)
        y_offset = self._draw_face_recognition_section(frame, dashboard_x, y_offset, face_result)
        
        # 4. Módulo de Fatiga
        y_offset = self._draw_fatigue_section(frame, dashboard_x, y_offset, fatigue_result)
        
        # 5. Módulo de Comportamientos
        y_offset = self._draw_behavior_section(frame, dashboard_x, y_offset, behavior_result)
        
        # 6. Estadísticas combinadas
        y_offset = self._draw_statistics_section(frame, dashboard_x, y_offset, fatigue_result, behavior_result, face_result)
        
        # 7. Alertas recientes
        y_offset = self._draw_alerts_section(frame, dashboard_x, y_offset)
        
        # Indicadores visuales en el video principal
        self._draw_video_overlays(frame, fatigue_result, behavior_result, face_result)
        
        # Renderizar AnalysisDashboard si está disponible
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
    
    def _draw_operator_section(self, frame, x, y, face_result):
        """Dibuja la sección de información del operador (actualizada)"""
        section_height = self.sections['operator']['height']
        
        # Título de sección
        cv2.putText(frame, "OPERADOR", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['text_secondary'], 1)
        
        # Línea separadora
        cv2.line(frame, (x + 10, y + 20), (x + self.width - 10, y + 20),
                self.colors['text_secondary'], 1)
        
        if face_result and face_result.get('operator_info'):
            operator_info = face_result['operator_info']
            
            # Nombre
            name = operator_info.get('name', 'Desconocido').replace('ñ', 'n')
            is_registered = operator_info.get('is_registered', False)
            
            # Color según estado
            name_color = self.colors['success'] if is_registered else self.colors['danger']
            
            cv2.putText(frame, f"Nombre: {name}", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       name_color, 1)
            
            # ID
            cv2.putText(frame, f"ID: {operator_info.get('id', 'N/A')}", (x + 15, y + 48),
                       self.font, self.font_sizes['small'],
                       self.colors['text_secondary'], 1)
            
            # Estado de registro
            if not is_registered:
                cv2.putText(frame, "NO REGISTRADO", (x + 15, y + 62),
                           self.font, self.font_sizes['body'],
                           self.colors['danger'], 2)
                
                # Tiempo como desconocido
                if hasattr(face_result, 'unknown_operator_time'):
                    unknown_time = face_result.get('unknown_operator_time', 0)
                    minutes = int(unknown_time / 60)
                    seconds = int(unknown_time % 60)
                    cv2.putText(frame, f"Tiempo: {minutes}:{seconds:02d}", (x + 150, y + 62),
                               self.font, self.font_sizes['small'],
                               self.colors['warning'], 1)
        else:
            cv2.putText(frame, "No identificado", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['warning'], 1)
        
        return y + section_height + 5
    
    def _draw_face_recognition_section(self, frame, x, y, result):
        """Dibuja la sección de reconocimiento facial (NUEVA)"""
        section_height = self.sections['face_recognition']['height']
        
        # Fondo de sección
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título con indicador
        has_unknown = False
        if result and result.get('operator_info'):
            has_unknown = not result['operator_info'].get('is_registered', True)
        
        status_color = self.colors['danger'] if has_unknown else self.colors['success']
        
        cv2.putText(frame, "RECONOCIMIENTO FACIAL", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   self.colors['face_color'], 1)
        
        cv2.circle(frame, (x + self.width - 20, y + 12), 4, status_color, -1)
        
        if not result:
            cv2.putText(frame, "Sin datos", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['text_secondary'], 1)
            return y + section_height + 5
        
        y_offset = y + 30
        
        # Confianza de reconocimiento
        if result.get('operator_info'):
            confidence = result['operator_info'].get('confidence', 0)
            conf_color = self._get_level_color(confidence * 100, [40, 60, 80])
            cv2.putText(frame, f"Confianza: {confidence:.2%}", (x + 15, y_offset),
                       self.font, self.font_sizes['body'],
                       conf_color, 1)
            
            # Barra de confianza
            y_offset += 15
            self._draw_progress_bar(frame, x + 15, y_offset, self.width - 30, 6,
                                   confidence, 0.7, conf_color, max_value=1.0)
        
        # Estado de calibración
        y_offset += 15
        is_calibrated = result.get('is_calibrated', False)
        calib_text = "Calibracion: PERSONALIZADA" if is_calibrated else "Calibracion: DEFAULT"
        calib_color = self.colors['success'] if is_calibrated else self.colors['warning']
        cv2.putText(frame, calib_text, (x + 15, y_offset),
                   self.font, self.font_sizes['small'],
                   calib_color, 1)
        
        # Mini gráfico de confianza
        y_offset += 20
        self._draw_mini_graph(frame, x + 15, y_offset, self.width - 30, 25,
                            self.face_confidence_history, "Historial confianza",
                            self.colors['face_color'])
        
        return y + section_height + 5
    
    def _draw_fatigue_section(self, frame, x, y, result):
        """Dibuja la sección de fatiga (versión compacta)"""
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
        
        # EAR compacto
        ear_value = result.get('ear_value', 0)
        ear_threshold = result.get('ear_threshold', 0.25)
        ear_color = self.colors['danger'] if ear_value < ear_threshold else self.colors['success']
        cv2.putText(frame, f"EAR: {ear_value:.3f}/{ear_threshold:.3f}", (x + 15, y_offset),
                   self.font, self.font_sizes['body'], ear_color, 1)
        
        # Nivel de fatiga
        y_offset += 20
        fatigue_pct = result.get('fatigue_percentage', 0)
        fatigue_color = self._get_level_color(fatigue_pct, [40, 60, 80])
        cv2.putText(frame, f"Fatiga: {fatigue_pct}%", (x + 15, y_offset),
                   self.font, self.font_sizes['body'], fatigue_color, 1)
        
        # Barra de fatiga
        y_offset += 15
        self._draw_progress_bar(frame, x + 15, y_offset, self.width - 30, 8,
                               fatigue_pct, 60, fatigue_color, max_value=100)
        
        # Microsueños
        y_offset += 20
        microsleeps = result.get('microsleep_count', 0)
        ms_color = self._get_level_color(microsleeps, [1, 2, 3])
        cv2.putText(frame, f"Microsuenos: {microsleeps}/3", (x + 15, y_offset),
                   self.font, self.font_sizes['body'], ms_color, 1)
        
        # Mini gráfico
        y_offset += 20
        self._draw_mini_graph(frame, x + 15, y_offset, self.width - 30, 30,
                            self.fatigue_history, "Historial", self.colors['fatigue_color'])
        
        return y + section_height + 5
    
    def _draw_behavior_section(self, frame, x, y, result):
        """Dibuja la sección de comportamientos (versión compacta)"""
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
        
        # Detecciones
        detections = result.get('detections', [])
        phone_detected = any(d[0] == 'cell phone' for d in detections)
        cigarette_detected = any(d[0] == 'cigarette' for d in detections)
        
        # Estado compacto
        phone_status = "TEL: SI" if phone_detected else "TEL: NO"
        phone_color = self.colors['danger'] if phone_detected else self.colors['text_secondary']
        
        cig_status = "CIG: SI" if cigarette_detected else "CIG: NO"
        cig_color = self.colors['warning'] if cigarette_detected else self.colors['text_secondary']
        
        cv2.putText(frame, phone_status, (x + 15, y_offset),
                   self.font, self.font_sizes['body'], phone_color, 1)
        cv2.putText(frame, cig_status, (x + 100, y_offset),
                   self.font, self.font_sizes['body'], cig_color, 1)
        
        # Duraciones si existen
        detector_info = result.get('detector_info', {})
        durations = detector_info.get('behavior_durations', {})
        
        if 'cell phone' in durations:
            y_offset += 20
            duration = durations['cell phone']
            cv2.putText(frame, f"Tel tiempo: {duration:.1f}s", (x + 15, y_offset),
                       self.font, self.font_sizes['small'],
                       self._get_level_color(duration, [3, 5, 7]), 1)
        
        # Mini gráfico
        y_offset = y + 100
        self._draw_mini_graph(frame, x + 15, y_offset, self.width - 30, 30,
                            self.behavior_history, "Actividad", self.colors['behavior_color'])
        
        return y + section_height + 5
    
    def _draw_statistics_section(self, frame, x, y, fatigue_result, behavior_result, face_result):
        """Dibuja estadísticas combinadas (actualizada)"""
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
        if face_result and face_result.get('operator_info') and not face_result['operator_info'].get('is_registered', True):
            status_parts.append("OPERADOR NO REGISTRADO")
        
        if not status_parts:
            status_text = "Estado: NORMAL"
            status_color = self.colors['success']
        else:
            status_text = "Estado: ALERTA"
            status_color = self.colors['danger']
        
        cv2.putText(frame, status_text, (x + 15, y_offset),
                   self.font, self.font_sizes['body'],
                   status_color, 1)
        
        # Módulos activos
        y_offset += 20
        active_modules = []
        if fatigue_result: active_modules.append("FAT")
        if behavior_result: active_modules.append("COMP")
        if face_result: active_modules.append("ROSTRO")
        
        modules_text = "Modulos: " + "/".join(active_modules)
        cv2.putText(frame, modules_text, (x + 15, y_offset),
                   self.font, self.font_sizes['small'],
                   self.colors['text_primary'], 1)
        
        # Tiempo de sesión
        y_offset += 20
        session_time = time.strftime("%M:%S", time.gmtime(time.time() % 3600))
        cv2.putText(frame, f"Tiempo: {session_time}", (x + 15, y_offset),
                   self.font, self.font_sizes['small'],
                   self.colors['text_secondary'], 1)
        
        return y + section_height + 5
    
    def _draw_alerts_section(self, frame, x, y):
        """Dibuja las alertas recientes (versión compacta)"""
        section_height = self.sections['alerts']['height']
        
        # Fondo de sección
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + section_height),
                     self.colors['section_bg'], -1)
        
        # Título
        alert_count = len(self.alert_history)
        title_color = self.colors['danger'] if alert_count > 0 else self.colors['text_secondary']
        cv2.putText(frame, f"ALERTAS ({alert_count})", (x + 10, y + 15),
                   self.font, self.font_sizes['section'],
                   title_color, 1)
        
        if not self.alert_history:
            cv2.putText(frame, "Sin alertas", (x + 15, y + 35),
                       self.font, self.font_sizes['body'],
                       self.colors['success'], 1)
            return y + section_height + 5
        
        # Mostrar últimas 3 alertas
        y_offset = y + 30
        for i, alert in enumerate(list(self.alert_history)[-3:]):
            alert_type = alert['type']
            time_ago = int(time.time() - alert['timestamp'])
            
            # Texto muy compacto
            if alert_type == 'fatigue':
                icon = "F"
                color = self.colors['fatigue_color']
            elif alert_type == 'behavior':
                icon = "C"
                color = self.colors['behavior_color']
            else:
                icon = "R"
                color = self.colors['face_color']
            
            text = f"[{icon}] {alert['message'][:20]}... ({time_ago}s)"
            cv2.putText(frame, text, (x + 15, y_offset),
                       self.font, self.font_sizes['small'],
                       color, 1)
            y_offset += 15
        
        return y + section_height + 5
    
    def _draw_video_overlays(self, frame, fatigue_result, behavior_result, face_result):
        """Dibuja overlays en el video principal (actualizado)"""
        h, w = frame.shape[:2]
        
        # Alerta de operador no registrado
        if face_result and face_result.get('operator_info'):
            operator_info = face_result['operator_info']
            if not operator_info.get('is_registered', True):
                # Marco rojo intermitente
                if int(time.time() * 2) % 2 == 0:  # Parpadea cada 0.5s
                    cv2.rectangle(frame, (2, 2), (w - 2, h - 2),
                                 self.colors['danger'], 3)
                
                # Mensaje de alerta
                msg = "OPERADOR NO REGISTRADO"
                text_size = cv2.getTextSize(msg, self.font, self.font_sizes['title'], 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, msg, (text_x, 30),
                           self.font, self.font_sizes['title'],
                           self.colors['danger'], 2)
        
        # Alertas de fatiga crítica
        elif fatigue_result and fatigue_result.get('is_critical', False):
            cv2.rectangle(frame, (2, 2), (w - 2, h - 2),
                         self.colors['danger'], 2)
            
            msg = "FATIGA CRITICA DETECTADA"
            text_size = cv2.getTextSize(msg, self.font, self.font_sizes['title'], 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, 60),
                       self.font, self.font_sizes['title'],
                       self.colors['danger'], 2)
        
        # Alertas de comportamiento
        elif behavior_result and behavior_result.get('alerts'):
            for alert in behavior_result['alerts']:
                if '7s' in alert[0]:
                    msg = "ALERTA: USO PROLONGADO TELEFONO"
                    cv2.putText(frame, msg, (10, 90),
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
    
    def _update_histories(self, fatigue_result, behavior_result, face_result):
        """Actualiza los historiales (actualizado)"""
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
        
        # Reconocimiento facial (NUEVO)
        if face_result and face_result.get('operator_info'):
            confidence = face_result['operator_info'].get('confidence', 0)
            self.face_confidence_history.append(confidence)
            
            # Alertas de operador no registrado
            if not face_result['operator_info'].get('is_registered', True):
                # Solo agregar alerta cada 30 segundos para no saturar
                if not any(a['type'] == 'face' and time.time() - a['timestamp'] < 30 
                          for a in self.alert_history):
                    self.alert_history.append({
                        'type': 'face',
                        'message': 'Operador no registrado',
                        'timestamp': time.time()
                    })
        else:
            self.face_confidence_history.append(0)