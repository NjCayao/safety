"""
Sistema Integrado de Detección de Distracciones
==============================================
Orquesta la detección de distracciones con calibración personalizada y reportes.
"""

import time
import logging
import os
import cv2
from collections import deque
from .distraction_detection import DistractionDetector
from .distraction_calibration import DistractionCalibration
from .distraction_dashboard import DistractionDashboard
from core.reports.report_manager import get_report_manager
from core.alarm_module import AlarmModule

class IntegratedDistractionSystem:
    def __init__(self, operators_dir="operators", dashboard_position='right'):
        """
        Inicializa el sistema integrado de detección de distracciones.
        
        Args:
            operators_dir: Directorio base de operadores
            dashboard_position: Posición del dashboard ('left' o 'right')
        """
        self.operators_dir = operators_dir
        self.logger = logging.getLogger('IntegratedDistractionSystem')
        
        # Componentes del sistema
        self.calibration_manager = DistractionCalibration(
            baseline_dir=os.path.join(operators_dir, "baseline-json")
        )
        
        # PRIMERO: Crear e inicializar AlarmModule
        self.alarm_module = AlarmModule(audio_dir="assets/audio")
        self.alarm_module.initialize()
        
        # DESPUÉS: Crear detector y conectar AlarmModule
        self.detector = DistractionDetector()
        self.detector.set_alarm_module(self.alarm_module)
        
        # Dashboard visual
        self.dashboard = DistractionDashboard(position=dashboard_position)
        self.dashboard_enabled = True
        
        # Gestor de reportes
        self.report_manager = get_report_manager()
        
        # Estado actual
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        
        # Control de distracciones
        self.distraction_times = deque()
        self.window_size = 600  # 10 minutos por defecto
        self.max_distractions_before_alert = 3
        
        # Control de reportes
        self.last_report_time = 0
        self.report_cooldown = 30  # segundos
        
        # Control de audio
        self.last_audio_time = 0
        self.audio_cooldown = 5  # segundos
        
        # Variables para captura mejorada
        self.current_distraction_frames = []  # Buffer de frames durante distracción
        self.max_rotation_during_distraction = 0
        self.frame_at_max_rotation = None
        self.distraction_start_time = None
        self.last_distraction_data = None
        self.extreme_event_recorded = False  # Para evitar múltiples registros del mismo evento
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_distractions': 0,
            'level1_events': 0,
            'level2_events': 0,
            'multiple_distraction_events': 0,
            'operators_monitored': set(),
            'reports_generated': 0
        }
        
        self.logger.info("Sistema integrado de distracciones inicializado con captura mejorada")
    
    def set_operator(self, operator_info):
        """
        Establece el operador actual y carga su calibración.
        
        Args:
            operator_info: Dict con 'id' y 'name' del operador
            
        Returns:
            bool: True si se estableció correctamente
        """
        if not operator_info or 'id' not in operator_info:
            self.logger.error("Información de operador inválida")
            return False
        
        operator_id = operator_info['id']
        
        # Si es el mismo operador, no hacer nada
        if self.current_operator and self.current_operator['id'] == operator_id:
            return True
        
        self.logger.info(f"Estableciendo operador: {operator_info.get('name', 'Unknown')} ({operator_id})")
        
        # Establecer nuevo operador
        self.current_operator = operator_info
        self.session_stats['operators_monitored'].add(operator_id)
        
        # Cargar calibración personalizada
        thresholds = self.calibration_manager.get_thresholds(operator_id)
        
        if thresholds.get('calibration_confidence', 0) > 0:
            self.is_calibrated = True
            self.logger.info(f"Calibración de distracciones cargada para {operator_id}")
        else:
            self.is_calibrated = False
            self.logger.warning(f"Sin calibración de distracciones para {operator_id}, usando valores por defecto")
        
        # Actualizar configuración
        self.current_thresholds = thresholds
        self.detector.update_config(thresholds)
        
        # Actualizar parámetros de control
        self.window_size = thresholds.get('distraction_window', 600)
        self.max_distractions_before_alert = 3  # Fijo en 3 como bostezos
        
        # Limpiar historiales
        self.distraction_times.clear()
        self.current_distraction_frames.clear()
        
        return True
    
    def analyze_frame(self, frame, landmarks):
        """
        Analiza un frame para detectar SOLO GIROS EXTREMOS.
        """
        if not self.current_operator:
            return {
                'status': 'error',
                'message': 'No hay operador activo',
                'frame': frame
            }
        
        # Incrementar contador
        self.session_stats['total_detections'] += 1
        
        # Detectar distracción (solo giros extremos)
        is_distracted, multiple_distractions = self.detector.detect(landmarks, frame)
        
        # Obtener estado del detector
        detector_status = self.detector.get_status()
        
        # Solo procesar si es GIRO EXTREMO
        if detector_status.get('direction') == 'EXTREMO':
            current_time = time.time()
            
            # Inicializar tiempo si es el inicio
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                self.logger.info("🔴 Inicio de GIRO EXTREMO detectado")
            
            # Verificar si alcanzó 7 segundos
            duration = current_time - self.distraction_start_time
            
            if duration >= 7.0 and not self.extreme_event_recorded:
                # Registrar evento de giro extremo
                self.distraction_times.append(current_time)
                self.session_stats['total_distractions'] += 1
                self.extreme_event_recorded = True
                
                # Capturar frame en este momento
                self.last_distraction_data = {
                    'frame': frame.copy(),
                    'timestamp': current_time,
                    'duration': duration
                }
                
                # Log
                count = len(self.distraction_times)
                self.logger.warning(f"⚠️ GIRO EXTREMO #{count} registrado (duración: {duration:.1f}s)")
                
                # NO reproducir audio aquí - el detector ya lo hace
        else:
            # Volvió a posición normal
            if self.distraction_start_time is not None:
                duration = time.time() - self.distraction_start_time
                if duration >= 3.0:  # Solo log si fue significativo
                    self.logger.info(f"✅ Volvió a posición normal después de {duration:.1f}s")
            
            # Resetear
            self.distraction_start_time = None
            self.extreme_event_recorded = False
        
        # Limpiar eventos antiguos (más de 10 minutos)
        self._clean_old_distractions()
        
        # Verificar si hay 3 giros extremos en 10 minutos
        total_extreme_events = len(self.distraction_times)
        should_report = total_extreme_events >= 3
        
        # Crear resultado
        result = {
            'is_distracted': detector_status.get('direction') == 'EXTREMO',
            'multiple_distractions': should_report,
            'detector_status': detector_status,
            'operator_info': self.current_operator,
            'operator_name': self.current_operator.get('name', 'Unknown'),
            'is_calibrated': self.is_calibrated,
            'total_distractions': total_extreme_events,
            'window_minutes': self.window_size // 60,
            'max_distractions': 3,
            'timestamp': time.time()
        }
        
        # Aplicar dashboard
        if self.dashboard_enabled:
            frame = self.dashboard.render(frame, result)
        
        result['frame'] = frame
        
        # Generar reporte si hay 3 giros extremos
        if should_report and not self.report_recently_sent():
            self._generate_extreme_rotation_report(result)
        
        return result
    
    def _generate_extreme_rotation_report(self, result):
        """Genera reporte por múltiples giros extremos"""
        current_time = time.time()
        
        # Verificar cooldown
        if current_time - self.last_report_time < self.report_cooldown:
            return
        
        self.logger.error(
            f"🚨 ALERTA CRÍTICA: {len(self.distraction_times)} giros extremos en {self.window_size//60} minutos - "
            f"Operador: {self.current_operator['name']}"
        )
        
        # Preparar datos
        event_data = {
            'extreme_rotation_count': len(self.distraction_times),
            'window_minutes': self.window_size // 60,
            'event_times': list(self.distraction_times),
            'operator_id': self.current_operator['id'],
            'operator_name': self.current_operator['name'],
            'severity': 'HIGH',
            'recommendation': 'Verificar estado del operador inmediatamente'
        }
        
        # Usar el frame capturado si está disponible
        frame_to_save = result['frame']
        if self.last_distraction_data and 'frame' in self.last_distraction_data:
            frame_to_save = self.last_distraction_data['frame']
            
            # Agregar texto de alerta
            cv2.putText(frame_to_save, 
                        "ALERTA: GIRO EXTREMO DETECTADO", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, 
                        (0, 0, 255), 
                        3)
        
        # Generar reporte
        report = self.report_manager.generate_report(
            module_name='distraction',
            event_type='multiple_extreme_rotations',
            data=event_data,
            frame=frame_to_save,
            operator_info=self.current_operator
        )
        
        if report:
            self.last_report_time = current_time
            self.session_stats['reports_generated'] += 1
            self.logger.info(f"📋 Reporte de giros extremos generado: {report['id']}")
    
    def _select_best_distraction_frame(self):
        """
        Selecciona el mejor frame de la distracción para capturar.
        Prioriza el frame con máxima rotación.
        """
        if not self.current_distraction_frames:
            return None
        
        # Estrategia principal: Frame con rotación máxima
        if self.frame_at_max_rotation:
            self.logger.debug(f"Usando frame con rotación máxima: {self.frame_at_max_rotation['rotation']:.1f}°")
            return self.frame_at_max_rotation
        
        # Estrategia alternativa: Frame en el punto medio temporal
        middle_index = len(self.current_distraction_frames) // 2
        middle_frame = self.current_distraction_frames[middle_index]
        self.logger.debug(f"Usando frame medio (índice {middle_index})")
        return middle_frame
    
    def _reset_distraction_buffers(self):
        """Limpia los buffers de captura para la próxima distracción"""
        self.max_rotation_during_distraction = 0
        self.frame_at_max_rotation = None
        self.current_distraction_frames.clear()
        self.distraction_start_time = None
    
    def _handle_distraction_detected(self, duration):
        """
        Maneja la detección de una distracción completa.
        
        Args:
            duration: Duración de la distracción en segundos
        """
        current_time = time.time()
        
        # Registrar distracción
        self.distraction_times.append(current_time)
        self.session_stats['total_distractions'] += 1
        
        # Contar evento según nivel
        if duration >= self.detector.config['level2_time']:
            self.session_stats['level2_events'] += 1
        else:
            self.session_stats['level1_events'] += 1
        
        # Seleccionar mejor frame
        best_frame_data = self._select_best_distraction_frame()
        
        if best_frame_data:
            # Crear información de captura
            capture_info = {
                'capture_rotation': best_frame_data['rotation'],
                'capture_direction': best_frame_data['direction'],
                'capture_time_offset': best_frame_data['duration_so_far'],
                'max_rotation_observed': self.max_rotation_during_distraction,
                'total_duration': duration,
                'frames_analyzed': len(self.current_distraction_frames),
                'validation_status': 'CONFIRMED_DISTRACTION'
            }
            
            self.logger.info(f"Distracción VALIDADA:")
            self.logger.info(f"  - Duración total: {duration:.2f}s")
            self.logger.info(f"  - Rotación máxima: {self.max_rotation_during_distraction:.1f}°")
            self.logger.info(f"  - Dirección: {best_frame_data['direction']}")
            
            # Guardar datos
            self.last_distraction_data = {
                'frame': best_frame_data['frame'],
                'capture_info': capture_info,
                'timestamp': current_time
            }
        
        # Log detallado
        distraction_count = len(self.distraction_times)
        self.logger.info(f"Distracción #{distraction_count} confirmada - Duración: {duration:.1f}s")
        
        # Reproducir audio según el número de distracciones
        if self.alarm_module and (current_time - self.last_audio_time > self.audio_cooldown):
            # Similar a bostezos: diferentes audios según número
            if distraction_count == 1:
                audio_key = "distraction"  # Primer audio
            elif distraction_count == 2:
                audio_key = "vadelante1"  # Segundo audio
            else:  # >= 3
                audio_key = "distraction"  # Tercer audio (más fuerte)
            
            self.logger.info(f"Reproduciendo: {audio_key}")
            success = self.alarm_module.play_audio(audio_key)
            
            if success:
                self.last_audio_time = current_time
    
    def _handle_multiple_distractions(self, result):
        """
        Maneja el evento de múltiples distracciones.
        Usa el frame capturado de la última distracción.
        """
        current_time = time.time()
        
        # Verificar cooldown de reporte
        if current_time - self.last_report_time < self.report_cooldown:
            return
        
        # Log de alerta
        self.logger.warning(
            f"ALERTA: {len(self.distraction_times)} distracciones en {self.window_size//60} minutos - "
            f"Operador: {self.current_operator['name']}"
        )
        
        # Incrementar contador de eventos
        self.session_stats['multiple_distraction_events'] += 1
        
        # Preparar datos del evento
        event_data = {
            'distraction_count': len(self.distraction_times),
            'window_minutes': self.window_size // 60,
            'max_allowed': self.max_distractions_before_alert,
            'distraction_times': list(self.distraction_times),
            'detector_status': result.get('detector_status', {}),
            'is_calibrated': self.is_calibrated,
            'analysis_timestamp': current_time
        }
        
        # Agregar información de captura si está disponible
        if self.last_distraction_data and self.last_distraction_data.get('capture_info'):
            event_data['capture_info'] = self.last_distraction_data['capture_info']
        
        # Usar el frame de la última distracción capturada si está disponible
        frame_to_save = result['frame']  # Frame actual con dashboard
        if self.last_distraction_data and 'frame' in self.last_distraction_data:
            # Aplicar dashboard al frame capturado
            frame_to_save = self.dashboard.render(self.last_distraction_data['frame'], result)
            
            # Agregar texto adicional
            cv2.putText(frame_to_save, 
                        f"Distracción capturada - Rotación: {self.last_distraction_data['capture_info']['capture_rotation']:.1f}°", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 255), 
                        2)
        
        # Generar reporte
        report = self.report_manager.generate_report(
            module_name='distraction',
            event_type='multiple_distractions',
            data=event_data,
            frame=frame_to_save,
            operator_info=self.current_operator
        )
        
        if report:
            self.last_report_time = current_time
            self.session_stats['reports_generated'] += 1
            self.logger.info(f"Reporte de múltiples distracciones generado: {report['id']}")
    
    def _clean_old_distractions(self):
        """Elimina distracciones fuera de la ventana temporal"""
        current_time = time.time()
        
        # Eliminar distracciones antiguas
        while self.distraction_times and (current_time - self.distraction_times[0] > self.window_size):
            removed_time = self.distraction_times.popleft()
            self.logger.debug(f"Distracción antigua eliminada (más de {self.window_size//60} minutos)")
    
    def force_distraction_report(self, frame):
        """
        Fuerza el envío de un reporte de múltiples distracciones (para testing).
        """
        if len(self.distraction_times) > 0:
            # Crear resultado artificial
            result = {
                'total_distractions': len(self.distraction_times),
                'window_minutes': self.window_size // 60,
                'max_distractions': self.max_distractions_before_alert,
                'multiple_distractions': True,
                'operator_info': self.current_operator,
                'detector_status': self.detector.get_status(),
                'frame': frame
            }
            
            self._handle_multiple_distractions(result)
            self.logger.warning("REPORTE FORZADO: Múltiples distracciones")
            return True
        else:
            self.logger.info("No hay distracciones para reportar")
            return False
    
    def reset_distraction_counter(self):
        """Reinicia el contador de distracciones"""
        self.distraction_times.clear()
        self._reset_distraction_buffers()
        self.logger.info("Contador de distracciones reiniciado")
        return True
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        return {
            'current_operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'detector_status': self.detector.get_status(),
            'distraction_count': len(self.distraction_times),
            'distraction_times': list(self.distraction_times),
            'session_stats': self.session_stats,
            'capture_buffer_size': len(self.current_distraction_frames)
        }
    
    def enable_dashboard(self, enabled=True):
        """Habilita o deshabilita el dashboard"""
        self.dashboard_enabled = enabled
    
    def set_dashboard_position(self, position):
        """Cambia la posición del dashboard"""
        if position in ['left', 'right']:
            self.dashboard.position = position
    
    def reset(self):
        """Reinicia el sistema completamente"""
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        self.distraction_times.clear()
        self.last_report_time = 0
        self.last_audio_time = 0
        self._reset_distraction_buffers()
        
        # Reiniciar estadísticas
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_distractions': 0,
            'level1_events': 0,
            'level2_events': 0,
            'multiple_distraction_events': 0,
            'operators_monitored': set(),
            'reports_generated': 0
        }
        
        # Reiniciar componentes
        self.detector = DistractionDetector()
        self.dashboard.reset()
        
        self.logger.info("Sistema de detección de distracciones reiniciado")