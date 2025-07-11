"""
Master Dashboard Profesional - Diseño Mejorado
=============================================
Dashboard elegante y profesional con diseño moderno y mejor legibilidad.
"""

import cv2
import numpy as np
import time
from collections import deque

class MasterDashboard:
    def __init__(self, width=350, position='left', enable_analysis_dashboard=True):
        """
        Inicializa el dashboard maestro con diseño profesional.
        
        Args:
            width: Ancho del panel del dashboard (default: 350px)
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
        
        # Colores del tema profesional (más suaves y elegantes)
        self.colors = {
            'background': (25, 25, 25),      # Gris oscuro más suave
            'panel_bg': (35, 35, 35),        # Gris panel
            'section_bg': (30, 30, 30),      # Gris sección
            'header_bg': (45, 45, 45),       # Gris encabezado
            'text_primary': (240, 240, 240), # Blanco suave
            'text_secondary': (160, 160, 160), # Gris claro
            'text_dim': (100, 100, 100),     # Gris muy suave
            'success': (46, 204, 113),       # Verde moderno
            'warning': (241, 196, 15),       # Amarillo elegante
            'danger': (231, 76, 60),         # Rojo moderno
            'info': (52, 152, 219),          # Azul informativo
            'accent': (155, 89, 182),        # Púrpura acento
            'graph_bg': (40, 40, 40),        # Fondo gráfico
            'graph_grid': (60, 60, 60),      # Líneas grid
            'divider': (50, 50, 50)          # Líneas divisoras
        }
        
        # Fuentes profesionales
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX
        self.font_sizes = {
            'header': 0.7,     # Título principal
            'title': 0.6,      # Títulos de sección
            'subtitle': 0.5,   # Subtítulos
            'body': 0.45,      # Texto normal
            'small': 0.4,      # Texto pequeño
            'tiny': 0.35       # Texto muy pequeño
        }
        
        # Iconos (representados con símbolos Unicode)
        self.icons = {
            'face': '◉',
            'fatigue': '◐',
            'behavior': '◈',
            'distraction': '◎',
            'yawn': '◑',
            'alert': '⚠',
            'check': '✓',
            'warning': '!',
            'info': 'i'
        }
        
        # Historiales para gráficos
        self.fatigue_history = deque(maxlen=50)
        self.behavior_history = deque(maxlen=50)
        self.face_confidence_history = deque(maxlen=50)
        self.distraction_history = deque(maxlen=50)
        self.yawn_history = deque(maxlen=50)
        self.alert_history = deque(maxlen=5)
        
        # Configuración de secciones con nuevo diseño
        self.sections = {
            'header': {'height': 60, 'padding': 10},
            'operator': {'height': 90, 'padding': 10},
            'modules': {'height': 280, 'padding': 10},  # Todos los módulos en una sección
            'statistics': {'height': 100, 'padding': 10},
            'alerts': {'height': 80, 'padding': 10}
        }
        
        # Estado de animaciones
        self.animation_counters = {}
        self.pulse_effect = 0
        
    def render(self, frame, fatigue_result=None, behavior_result=None, face_result=None, 
               distraction_result=None, yawn_result=None, analysis_data=None):
        """
        Renderiza el dashboard con diseño profesional.
        """
        h, w = frame.shape[:2]
        
        # Determinar posición X del dashboard
        if self.position == 'left':
            dashboard_x = self.margin
        else:
            dashboard_x = w - self.width - self.margin
        
        # Crear overlay para el dashboard con bordes redondeados simulados
        overlay = frame.copy()
        
        # Fondo principal con efecto de sombra
        self._draw_panel_background(overlay, dashboard_x, self.margin, 
                                   self.width, h - 2*self.margin)
        
        # Aplicar transparencia elegante
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        # Actualizar historiales y animaciones
        self._update_histories(fatigue_result, behavior_result, face_result, 
                             distraction_result, yawn_result)
        self.pulse_effect = (self.pulse_effect + 5) % 360
        
        # Renderizar secciones con nuevo diseño
        y_offset = self.margin + 10
        
        # 1. Header elegante
        y_offset = self._draw_header_section(frame, dashboard_x, y_offset)
        
        # 2. Información del operador mejorada
        y_offset = self._draw_operator_card(frame, dashboard_x, y_offset, face_result)
        
        # 3. Módulos en grid compacto
        y_offset = self._draw_modules_grid(frame, dashboard_x, y_offset,
                                         fatigue_result, behavior_result, face_result,
                                         distraction_result, yawn_result)
        
        # 4. Estadísticas resumidas
        y_offset = self._draw_statistics_summary(frame, dashboard_x, y_offset,
                                               fatigue_result, behavior_result, face_result,
                                               distraction_result, yawn_result)
        
        # 5. Centro de alertas
        y_offset = self._draw_alert_center(frame, dashboard_x, y_offset)
        
        # Indicadores visuales mejorados en el video
        self._draw_elegant_overlays(frame, fatigue_result, behavior_result, face_result,
                                   distraction_result, yawn_result)
        
        # Renderizar AnalysisDashboard si está disponible
        if self.analysis_dashboard and analysis_data:
            frame = self.analysis_dashboard.render(frame, analysis_data)
        
        return frame
    
    def _draw_panel_background(self, frame, x, y, width, height):
        """Dibuja el fondo del panel con efecto de profundidad"""
        # Sombra
        shadow_offset = 3
        cv2.rectangle(frame, 
                     (x + shadow_offset, y + shadow_offset), 
                     (x + width + shadow_offset, y + height + shadow_offset),
                     (0, 0, 0), -1)
        
        # Panel principal
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['background'], -1)
        
        # Borde sutil
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['divider'], 1)
    
    def _draw_header_section(self, frame, x, y):
        """Dibuja el header con diseño elegante"""
        height = self.sections['header']['height']
        
        # Fondo del header con gradiente simulado
        cv2.rectangle(frame, (x + 5, y), (x + self.width - 5, y + height),
                     self.colors['header_bg'], -1)
        
        # Título principal centrado
        title = "SISTEMA DE MONITOREO"
        subtitle = "Control Integral de Operadores"
        
        # Título con efecto
        text_size = cv2.getTextSize(title, self.font_bold, self.font_sizes['header'], 2)[0]
        text_x = x + (self.width - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, y + 25),
                   self.font_bold, self.font_sizes['header'],
                   self.colors['text_primary'], 2)
        
        # Subtítulo
        text_size = cv2.getTextSize(subtitle, self.font, self.font_sizes['small'], 1)[0]
        text_x = x + (self.width - text_size[0]) // 2
        cv2.putText(frame, subtitle, (text_x, y + 42),
                   self.font, self.font_sizes['small'],
                   self.colors['text_secondary'], 1)
        
        # Hora con formato elegante
        time_str = time.strftime("%H:%M:%S")
        date_str = time.strftime("%d/%m/%Y")
        
        # Contenedor de tiempo
        time_box_width = 80
        cv2.rectangle(frame, 
                     (x + self.width - time_box_width - 15, y + 5),
                     (x + self.width - 10, y + 30),
                     self.colors['section_bg'], -1)
        
        cv2.putText(frame, time_str, (x + self.width - time_box_width - 5, y + 20),
                   self.font, self.font_sizes['small'],
                   self.colors['text_primary'], 1)
        
        # Línea divisora elegante
        self._draw_gradient_line(frame, x + 20, y + height - 5, 
                                self.width - 40, self.colors['accent'])
        
        return y + height + 10
    
    def _draw_operator_card(self, frame, x, y, face_result):
        """Dibuja la tarjeta del operador con diseño mejorado"""
        height = self.sections['operator']['height']
        padding = 15
        
        # Fondo de la tarjeta
        cv2.rectangle(frame, (x + padding, y), 
                     (x + self.width - padding, y + height),
                     self.colors['section_bg'], -1)
        
        # Título de sección con icono
        self._draw_section_title(frame, "OPERADOR", x + padding + 10, y + 20,
                                self.colors['info'])
        
        if face_result and face_result.get('operator_info'):
            operator_info = face_result['operator_info']
            is_registered = operator_info.get('is_registered', False)
            
            # Foto placeholder o inicial
            photo_size = 50
            photo_x = x + padding + 15
            photo_y = y + 35
            
            # Marco de foto
            if is_registered:
                cv2.circle(frame, (photo_x + photo_size//2, photo_y + photo_size//2),
                          photo_size//2 + 2, self.colors['success'], 2)
            else:
                cv2.circle(frame, (photo_x + photo_size//2, photo_y + photo_size//2),
                          photo_size//2 + 2, self.colors['danger'], 2)
            
            # Inicial en el círculo
            name = operator_info.get('name', 'Desconocido')
            initial = name[0].upper() if name != 'Desconocido' else '?'
            text_size = cv2.getTextSize(initial, self.font_bold, 0.8, 2)[0]
            cv2.putText(frame, initial,
                       (photo_x + photo_size//2 - text_size[0]//2,
                        photo_y + photo_size//2 + text_size[1]//2),
                       self.font_bold, 0.8, self.colors['text_primary'], 2)
            
            # Información del operador
            info_x = photo_x + photo_size + 15
            
            # Nombre
            cv2.putText(frame, name, (info_x, photo_y + 15),
                       self.font_bold, self.font_sizes['subtitle'],
                       self.colors['text_primary'], 1)
            
            # ID con formato
            id_text = f"ID: {operator_info.get('id', 'N/A')}"
            cv2.putText(frame, id_text, (info_x, photo_y + 35),
                       self.font, self.font_sizes['small'],
                       self.colors['text_secondary'], 1)
            
            # Estado con badge
            if is_registered:
                self._draw_badge(frame, info_x, photo_y + 45, "REGISTRADO", 
                               self.colors['success'], 80)
            else:
                self._draw_badge(frame, info_x, photo_y + 45, "NO REGISTRADO", 
                               self.colors['danger'], 100)
        else:
            # Sin operador detectado
            cv2.putText(frame, "Esperando detección...", 
                       (x + self.width//2 - 70, y + height//2),
                       self.font, self.font_sizes['body'],
                       self.colors['text_dim'], 1)
        
        return y + height + 10
    
    def _draw_modules_grid(self, frame, x, y, fatigue_result, behavior_result, 
                          face_result, distraction_result, yawn_result):
        """Dibuja los módulos en un grid compacto y elegante"""
        height = self.sections['modules']['height']
        padding = 15
        
        # Fondo de la sección
        cv2.rectangle(frame, (x + padding, y), 
                     (x + self.width - padding, y + height),
                     self.colors['section_bg'], -1)
        
        # Título
        self._draw_section_title(frame, "MODULOS ACTIVOS", x + padding + 10, y + 20,
                                self.colors['accent'])
        
        # Grid 2x3 para los módulos
        module_width = (self.width - 2*padding - 20) // 2
        module_height = 75
        spacing = 10
        start_y = y + 40
        
        modules = [
            ("RECONOCIMIENTO", face_result, self.colors['info'], 
             self._get_face_status(face_result)),
            ("MICROSUENO", fatigue_result, self.colors['warning'],
             self._get_fatigue_status(fatigue_result)),
            ("CELULAR/CIGARRO", behavior_result, self.colors['danger'],
             self._get_behavior_status(behavior_result)),
            ("DISTRACCIONES", distraction_result, self.colors['warning'],
             self._get_distraction_status(distraction_result)),
            ("BOSTEZOS", yawn_result, self.colors['info'],
             self._get_yawn_status(yawn_result))
        ]
        
        for i, (name, result, color, status) in enumerate(modules):
            row = i // 2
            col = i % 2
            
            mod_x = x + padding + 10 + (module_width + spacing) * col
            mod_y = start_y + (module_height + spacing) * row
            
            self._draw_module_card(frame, mod_x, mod_y, module_width, module_height,
                                  name, status, color, result is not None)
        
        return y + height + 10
    
    def _draw_module_card(self, frame, x, y, width, height, name, status, color, active):
        """Dibuja una tarjeta de módulo individual"""
        # Fondo con borde
        bg_color = self.colors['panel_bg'] if active else self.colors['background']
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     color if active else self.colors['divider'], 1)
        
        # Indicador de estado (círculo)
        indicator_x = x + width - 15
        indicator_y = y + 15
        cv2.circle(frame, (indicator_x, indicator_y), 5,
                  color if active and status['active'] else self.colors['text_dim'], -1)
        
        # Nombre del módulo
        cv2.putText(frame, name, (x + 10, y + 20),
                   self.font, self.font_sizes['small'],
                   self.colors['text_primary'] if active else self.colors['text_dim'], 1)
        
        # Estado principal
        if active:
            cv2.putText(frame, status['text'], (x + 10, y + 40),
                       self.font_bold, self.font_sizes['body'],
                       status['color'], 1)
            
            # Información adicional
            if status['info']:
                cv2.putText(frame, status['info'], (x + 10, y + 58),
                           self.font, self.font_sizes['tiny'],
                           self.colors['text_secondary'], 1)
    
    def _draw_statistics_summary(self, frame, x, y, *results):
        """Dibuja resumen de estadísticas con diseño limpio"""
        height = self.sections['statistics']['height']
        padding = 15
        
        # Fondo
        cv2.rectangle(frame, (x + padding, y), 
                     (x + self.width - padding, y + height),
                     self.colors['section_bg'], -1)
        
        # Título
        self._draw_section_title(frame, "RESUMEN DE SESION", x + padding + 10, y + 20,
                                self.colors['text_secondary'])
        
        # Calcular estadísticas
        stats = self._calculate_session_stats(*results)
        
        # Mostrar en dos columnas
        col1_x = x + padding + 15
        col2_x = x + self.width//2 + 10
        stat_y = y + 40
        
        # Columna 1
        cv2.putText(frame, f"Alertas totales: {stats['total_alerts']}", 
                   (col1_x, stat_y),
                   self.font, self.font_sizes['small'],
                   self.colors['text_primary'], 1)
        
        cv2.putText(frame, f"Estado general: {stats['overall_status']}", 
                   (col1_x, stat_y + 20),
                   self.font, self.font_sizes['small'],
                   stats['status_color'], 1)
        
        # Columna 2
        cv2.putText(frame, f"Modulos OK: {stats['modules_ok']}/5", 
                   (col2_x, stat_y),
                   self.font, self.font_sizes['small'],
                   self.colors['text_primary'], 1)
        
        cv2.putText(frame, f"Tiempo: {stats['session_time']}", 
                   (col2_x, stat_y + 20),
                   self.font, self.font_sizes['small'],
                   self.colors['text_secondary'], 1)
        
        # Mini barra de progreso de salud general
        self._draw_health_bar(frame, x + padding + 15, y + height - 20,
                            self.width - 2*padding - 30, 10, stats['health_score'])
        
        return y + height + 10
    
    def _draw_alert_center(self, frame, x, y):
        """Dibuja el centro de alertas con diseño moderno"""
        height = self.sections['alerts']['height']
        padding = 15
        
        # Fondo
        cv2.rectangle(frame, (x + padding, y), 
                     (x + self.width - padding, y + height),
                     self.colors['section_bg'], -1)
        
        # Título con contador
        alert_count = len(self.alert_history)
        title_color = self.colors['danger'] if alert_count > 0 else self.colors['text_secondary']
        self._draw_section_title(frame, f"CENTRO DE ALERTAS ({alert_count})", 
                                x + padding + 10, y + 20, title_color)
        
        if not self.alert_history:
            # Sin alertas - mostrar mensaje positivo
            cv2.putText(frame, "Sistema operando normalmente", 
                       (x + padding + 20, y + 45),
                       self.font, self.font_sizes['body'],
                       self.colors['success'], 1)
        else:
            # Mostrar alertas con diseño mejorado
            alert_y = y + 35
            for i, alert in enumerate(list(self.alert_history)[-2:]):  # Solo últimas 2
                self._draw_alert_item(frame, x + padding + 15, alert_y, 
                                    self.width - 2*padding - 30, alert)
                alert_y += 20
        
        return y + height + 10
    
    def _draw_alert_item(self, frame, x, y, width, alert):
        """Dibuja un item de alerta individual"""
        # Icono según tipo
        icon_map = {
            'fatigue': ('◐', self.colors['warning']),
            'behavior': ('◈', self.colors['danger']),
            'face': ('◉', self.colors['info']),
            'distraction': ('◎', self.colors['warning']),
            'yawn': ('◑', self.colors['info'])
        }
        
        icon, color = icon_map.get(alert['type'], ('!', self.colors['danger']))
        
        # Icono
        cv2.putText(frame, icon, (x, y + 12),
                   self.font, self.font_sizes['body'],
                   color, 1)
        
        # Mensaje
        time_ago = int(time.time() - alert['timestamp'])
        msg = f"{alert['message'][:30]}... ({time_ago}s)"
        cv2.putText(frame, msg, (x + 20, y + 12),
                   self.font, self.font_sizes['tiny'],
                   self.colors['text_primary'], 1)
    
    def _draw_elegant_overlays(self, frame, *results):
        """Dibuja overlays elegantes en el video principal"""
        h, w = frame.shape[:2]
        
        # Detectar condición crítica
        critical_alert = None
        
        # Verificar cada resultado
        if results[2] and results[2].get('operator_info'):  # face_result
            if not results[2]['operator_info'].get('is_registered', True):
                critical_alert = ("OPERADOR NO REGISTRADO", self.colors['danger'])
        
        elif results[0] and results[0].get('is_critical'):  # fatigue_result
            critical_alert = ("ALERTA: FATIGA CRÍTICA", self.colors['danger'])
        
        elif results[3] and results[3].get('detector_status', {}).get('total_distractions', 0) >= 3:
            critical_alert = ("ALERTA: MÚLTIPLES DISTRACCIONES", self.colors['warning'])
        
        # Mostrar alerta crítica si existe
        if critical_alert:
            # Banner superior elegante
            banner_height = 40
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Texto centrado
            text, color = critical_alert
            text_size = cv2.getTextSize(text, self.font_bold, self.font_sizes['title'], 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, 25),
                       self.font_bold, self.font_sizes['title'],
                       color, 2)
    
    # Métodos auxiliares
    def _draw_section_title(self, frame, title, x, y, color):
        """Dibuja un título de sección con estilo"""
        cv2.putText(frame, title, (x, y),
                   self.font_bold, self.font_sizes['subtitle'],
                   color, 1)
    
    def _draw_badge(self, frame, x, y, text, color, width):
        """Dibuja un badge/etiqueta"""
        height = 20
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
        
        text_size = cv2.getTextSize(text, self.font, self.font_sizes['tiny'], 1)[0]
        text_x = x + (width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, y + 14),
                   self.font, self.font_sizes['tiny'],
                   self.colors['text_primary'], 1)
    
    def _draw_gradient_line(self, frame, x, y, width, color):
        """Dibuja una línea con efecto gradiente"""
        for i in range(width):
            fade = 1.0 - abs(i - width/2) / (width/2)
            line_color = tuple(int(c * fade) for c in color)
            cv2.line(frame, (x + i, y), (x + i, y + 1), line_color, 1)
    
    def _draw_health_bar(self, frame, x, y, width, height, score):
        """Dibuja una barra de salud general"""
        # Fondo
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['graph_bg'], -1)
        
        # Valor
        fill_width = int(width * score / 100)
        if score > 70:
            color = self.colors['success']
        elif score > 40:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Texto
        cv2.putText(frame, f"{score}%", (x + width//2 - 15, y - 3),
                   self.font, self.font_sizes['tiny'],
                   self.colors['text_secondary'], 1)
    
    # Métodos de estado para cada módulo
    def _get_face_status(self, result):
        if not result or not result.get('operator_info'):
            return {'active': False, 'text': 'Sin detección', 
                   'color': self.colors['text_dim'], 'info': None}
        
        info = result['operator_info']
        confidence = info.get('confidence', 0)
        is_registered = info.get('is_registered', True)
        
        if not is_registered:
            return {'active': True, 'text': 'NO REGISTRADO',
                   'color': self.colors['danger'], 
                   'info': f'Confianza: {confidence:.0%}'}
        
        return {'active': True, 'text': 'IDENTIFICADO',
               'color': self.colors['success'],
               'info': f'Confianza: {confidence:.0%}'}
    
    def _get_fatigue_status(self, result):
        if not result:
            return {'active': False, 'text': 'Inactivo',
                   'color': self.colors['text_dim'], 'info': None}
        
        fatigue_pct = result.get('fatigue_percentage', 0)
        microsleeps = result.get('microsleep_count', 0)
        
        if result.get('is_critical'):
            return {'active': True, 'text': 'CRÍTICO',
                   'color': self.colors['danger'],
                   'info': f'Fatiga: {fatigue_pct}% | MS: {microsleeps}'}
        elif microsleeps > 0:
            return {'active': True, 'text': 'ALERTA',
                   'color': self.colors['warning'],
                   'info': f'Fatiga: {fatigue_pct}% | MS: {microsleeps}'}
        
        return {'active': True, 'text': 'NORMAL',
               'color': self.colors['success'],
               'info': f'Fatiga: {fatigue_pct}%'}
    
    def _get_behavior_status(self, result):
        if not result:
            return {'active': False, 'text': 'Inactivo',
                   'color': self.colors['text_dim'], 'info': None}
        
        detections = result.get('detections', [])
        alerts = result.get('alerts', [])
        
        phone = any(d[0] == 'cell phone' for d in detections)
        cigarette = any(d[0] == 'cigarette' for d in detections)
        
        if alerts:
            return {'active': True, 'text': 'ALERTA',
                   'color': self.colors['danger'],
                   'info': f'Tel: {"Sí" if phone else "No"} | Cig: {"Sí" if cigarette else "No"}'}
        elif detections:
            return {'active': True, 'text': 'DETECTADO',
                   'color': self.colors['warning'],
                   'info': f'Tel: {"Sí" if phone else "No"} | Cig: {"Sí" if cigarette else "No"}'}
        
        return {'active': True, 'text': 'NORMAL',
               'color': self.colors['success'],
               'info': 'Sin detecciones'}
    
    def _get_distraction_status(self, result):
        if not result or not result.get('detector_status'):
            return {'active': False, 'text': 'Inactivo',
                   'color': self.colors['text_dim'], 'info': None}
        
        status = result['detector_status']
        direction = status.get('direction', 'CENTRO')
        count = status.get('total_distractions', 0)
        
        if direction == 'EXTREMO':
            return {'active': True, 'text': 'GIRO EXTREMO',
                   'color': self.colors['danger'],
                   'info': f'Eventos: {count}/3'}
        elif direction == 'SIN ROSTRO':
            return {'active': True, 'text': 'SIN ROSTRO',
                   'color': self.colors['warning'],
                   'info': None}
        elif direction == 'AUSENTE':
            return {'active': True, 'text': 'AUSENTE',
                   'color': self.colors['text_dim'],
                   'info': None}
        
        return {'active': True, 'text': 'CENTRO',
               'color': self.colors['success'],
               'info': f'Eventos: {count}/3'}
    
    def _get_yawn_status(self, result):
        if not result:
            return {'active': False, 'text': 'Inactivo',
                   'color': self.colors['text_dim'], 'info': None}
        
        detection = result.get('detection_result', {})
        is_yawning = detection.get('is_yawning', False)
        count = result.get('yawn_count', 0)
        
        if count >= 3:
            return {'active': True, 'text': 'MÚLTIPLES',
                   'color': self.colors['danger'],
                   'info': f'Bostezos: {count}/3'}
        elif is_yawning:
            return {'active': True, 'text': 'BOSTEZANDO',
                   'color': self.colors['warning'],
                   'info': f'Bostezos: {count}/3'}
        
        return {'active': True, 'text': 'NORMAL',
               'color': self.colors['success'],
               'info': f'Bostezos: {count}/3'}
    
    def _calculate_session_stats(self, *results):
        """Calcula estadísticas de la sesión"""
        total_alerts = 0
        modules_ok = 0
        health_score = 100
        
        # Contar alertas y módulos OK
        for result in results:
            if result:
                modules_ok += 1
                
                # Verificar alertas específicas
                if isinstance(result, dict):
                    if result.get('is_critical'):
                        total_alerts += 1
                        health_score -= 20
                    elif result.get('alerts'):
                        total_alerts += len(result.get('alerts', []))
                        health_score -= 10
                    elif result.get('detector_status', {}).get('is_distracted'):
                        total_alerts += 1
                        health_score -= 15
        
        # Estado general
        if health_score > 70:
            overall_status = "OPTIMO"
            status_color = self.colors['success']
        elif health_score > 40:
            overall_status = "ACEPTABLE"
            status_color = self.colors['warning']
        else:
            overall_status = "CRITICO"
            status_color = self.colors['danger']
        
        # Tiempo de sesión
        session_time = time.strftime("%M:%S", time.gmtime(time.time() % 3600))
        
        return {
            'total_alerts': total_alerts,
            'modules_ok': modules_ok,
            'health_score': max(0, health_score),
            'overall_status': overall_status,
            'status_color': status_color,
            'session_time': session_time
        }
    
    def _update_histories(self, fatigue_result, behavior_result, face_result, 
                        distraction_result, yawn_result):
        """Actualiza los historiales"""
        # Fatiga
        if fatigue_result:
            self.fatigue_history.append(fatigue_result.get('fatigue_percentage', 0) / 100)
            
            if fatigue_result.get('microsleep_detected', False):
                self.alert_history.append({
                    'type': 'fatigue',
                    'message': 'Microsueño detectado',
                    'timestamp': time.time()
                })
        
        # Comportamiento
        if behavior_result:
            has_detection = len(behavior_result.get('detections', [])) > 0
            self.behavior_history.append(1.0 if has_detection else 0.0)
            
            for alert in behavior_result.get('alerts', []):
                self.alert_history.append({
                    'type': 'behavior',
                    'message': self._format_behavior_alert(alert[0]),
                    'timestamp': time.time()
                })
        
        # Reconocimiento facial
        if face_result and face_result.get('operator_info'):
            confidence = face_result['operator_info'].get('confidence', 0)
            self.face_confidence_history.append(confidence)
            
            if not face_result['operator_info'].get('is_registered', True):
                if not any(a['type'] == 'face' and time.time() - a['timestamp'] < 30 
                          for a in self.alert_history):
                    self.alert_history.append({
                        'type': 'face',
                        'message': 'Operador no registrado',
                        'timestamp': time.time()
                    })
        else:
            self.face_confidence_history.append(0)
        
        # Distracciones
        if distraction_result and 'detector_status' in distraction_result:
            detector_status = distraction_result['detector_status']
            is_distracted = detector_status.get('is_distracted', False)
            self.distraction_history.append(1.0 if is_distracted else 0.0)
            
            if detector_status.get('total_distractions', 0) >= 3:
                if not any(a['type'] == 'distraction' and time.time() - a['timestamp'] < 30
                          for a in self.alert_history):
                    self.alert_history.append({
                        'type': 'distraction',
                        'message': 'Múltiples giros extremos',
                        'timestamp': time.time()
                    })
        else:
            self.distraction_history.append(0)
        
        # Bostezos
        if yawn_result and 'detection_result' in yawn_result:
            detection = yawn_result['detection_result']
            is_yawning = detection.get('is_yawning', False)
            self.yawn_history.append(1.0 if is_yawning else 0.0)
            
            if yawn_result.get('yawn_count', 0) >= 3:
                if not any(a['type'] == 'yawn' and time.time() - a['timestamp'] < 30
                          for a in self.alert_history):
                    self.alert_history.append({
                        'type': 'yawn',
                        'message': 'Múltiples bostezos detectados',
                        'timestamp': time.time()
                    })
        else:
            self.yawn_history.append(0)
    
    def _format_behavior_alert(self, alert_type):
        """Formatea mensajes de alerta de comportamiento"""
        formats = {
            'phone_3s': 'Uso de teléfono (3s)',
            'phone_7s': 'Uso prolongado teléfono',
            'smoking_pattern': 'Patrón de fumar',
            'smoking_7s': 'Fumando continuamente'
        }
        return formats.get(alert_type, alert_type)