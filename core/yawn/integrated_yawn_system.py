"""
Sistema Integrado de Detección de Bostezos - Versión Mejorada
===========================================================
Captura el frame en el punto máximo del bostezo y valida completamente.
"""

import time
import logging
import os
import pygame
import cv2
from collections import deque
from .yawn_detection import YawnDetector
from .yawn_calibration import YawnCalibration
from .yawn_dashboard import YawnDashboard
from core.reports.report_manager import get_report_manager
from core.alarm_module import AlarmModule

class IntegratedYawnSystem:
    def __init__(self, operators_dir="operators", dashboard_position='right'):
        """
        Inicializa el sistema integrado de detección de bostezos.
        
        Args:
            operators_dir: Directorio base de operadores
            dashboard_position: Posición del dashboard ('left' o 'right')
        """
        self.operators_dir = operators_dir
        self.logger = logging.getLogger('IntegratedYawnSystem')
        
        # Componentes del sistema
        self.calibration_manager = YawnCalibration(
            baseline_dir=os.path.join(operators_dir, "baseline-json")
        )
        
        # Detector de bostezos
        self.detector = YawnDetector()
        
        # Dashboard visual
        self.dashboard = YawnDashboard(position=dashboard_position)
        self.dashboard_enabled = True
        
        # Gestor de reportes
        self.report_manager = get_report_manager()
        
        # Estado actual
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        
        # Control de bostezos
        self.yawn_times = deque()
        self.window_size = 600  # 10 minutos por defecto
        self.max_yawns_before_alert = 3
        
        # Control de reportes
        self.last_report_time = 0
        self.report_cooldown = 30  # segundos
        
        # Control de audio
        self.last_audio_time = 0
        self.audio_cooldown = 5  # segundos
        self.alarm_module = AlarmModule(audio_dir="assets/audio")
        self.alarm_module.initialize()
        
        # === NUEVO: Variables para captura mejorada ===
        self.current_yawn_frames = []  # Buffer de frames durante el bostezo
        self.max_mar_during_yawn = 0
        self.frame_at_max_mar = None
        self.yawn_start_time = None
        self.last_yawn_data = None  # Datos del último bostezo para reportes
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_yawns': 0,
            'multiple_yawn_events': 0,
            'operators_monitored': set(),
            'reports_generated': 0
        }
        
        self.logger.info("Sistema integrado de bostezos inicializado con captura mejorada")
    
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
            self.logger.info(f"Calibración de bostezos cargada para {operator_id}")
        else:
            self.is_calibrated = False
            self.logger.warning(f"Sin calibración de bostezos para {operator_id}, usando valores por defecto")
        
        # Actualizar configuración
        self.current_thresholds = thresholds
        self.detector.update_config(thresholds)
        
        # Actualizar parámetros de control
        self.window_size = thresholds.get('window_size', 600)
        self.max_yawns_before_alert = thresholds.get('max_yawns_before_alert', 3)
        
        # Limpiar historiales
        self.yawn_times.clear()
        self.current_yawn_frames.clear()
        
        return True
    
    def analyze_frame(self, frame, landmarks):
        """
        Analiza un frame para detectar bostezos con captura mejorada.
        ACTUALIZADO: Captura el frame CON los dibujos de contorno y puntos
        """
        if not self.current_operator:
            return {
                'status': 'error',
                'message': 'No hay operador activo',
                'frame': frame
            }
        
        # Incrementar contador
        self.session_stats['total_detections'] += 1
        
        # Detectar bostezo
        detection_result = self.detector.detect(frame, landmarks)
        
        # === IMPORTANTE: Dibujar información ANTES de guardar ===
        # Crear una copia del frame con los dibujos
        frame_with_drawings = self.detector.draw_yawn_info(frame.copy(), detection_result)
        
        # Si está bostezando, guardar frames CON DIBUJOS
        if detection_result.get('is_yawning', False):
            current_mar = detection_result.get('mar_value', 0)
            current_time = time.time()
            
            # Inicializar tiempo si es el inicio
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
                self.logger.debug("Inicio de bostezo detectado para captura")
            
            # Crear datos del frame CON DIBUJOS
            frame_data = {
                'frame': frame_with_drawings.copy(),  # ← CAMBIO: Usar frame con dibujos
                'frame_original': frame.copy(),       # ← NUEVO: Guardar también el original
                'mar': current_mar,
                'timestamp': current_time,
                'duration_so_far': current_time - self.yawn_start_time
            }
            
            # Mantener buffer limitado (3 segundos a 30fps)
            if len(self.current_yawn_frames) >= 90:
                self.current_yawn_frames.pop(0)
            self.current_yawn_frames.append(frame_data)
            
            # Actualizar máximo MAR
            if current_mar > self.max_mar_during_yawn:
                self.max_mar_during_yawn = current_mar
                self.frame_at_max_mar = frame_data
                self.logger.debug(f"Nuevo MAR máximo: {current_mar:.3f} a {frame_data['duration_so_far']:.2f}s")
        
        # Guardar estado antes de procesar
        yawn_count_before = len(self.yawn_times)
        
        # Cuando termina el bostezo, validar y procesar
        if detection_result.get('yawn_detected', False):
            final_duration = detection_result.get('yawn_duration', 0)
            
            # VALIDACIÓN: Solo si es bostezo real
            if final_duration >= self.detector.config['duration_threshold']:
                # Seleccionar el mejor frame
                best_frame_data = self._select_best_yawn_frame()
                
                if best_frame_data:
                    # Crear información de captura
                    capture_info = {
                        'capture_mar': best_frame_data['mar'],
                        'capture_time_offset': best_frame_data['duration_so_far'],
                        'max_mar_observed': self.max_mar_during_yawn,
                        'total_duration': final_duration,
                        'frames_analyzed': len(self.current_yawn_frames),
                        'validation_status': 'CONFIRMED_REAL_YAWN'
                    }
                    
                    self.logger.info(f"Bostezo VALIDADO:")
                    self.logger.info(f"  - Duración total: {final_duration:.2f}s")
                    self.logger.info(f"  - MAR máximo: {self.max_mar_during_yawn:.3f}")
                    self.logger.info(f"  - Captura tomada a: {capture_info['capture_time_offset']:.2f}s del inicio")
                    
                    # Procesar con el frame seleccionado
                    self._handle_yawn_detected(detection_result, best_frame_data['frame'], capture_info)
                else:
                    # Sin frame válido, usar el actual
                    self.logger.warning("No hay frames en buffer, usando frame actual")
                    self._handle_yawn_detected(detection_result, frame, None)
            else:
                self.logger.debug(f"Bostezo descartado (duración: {final_duration:.2f}s < {self.detector.config['duration_threshold']}s)")
            
            # Limpiar buffers para el próximo bostezo
            self._reset_yawn_buffers()
        
        # Limpiar bostezos antiguos
        self._clean_old_yawns()
        
        # Verificar múltiples bostezos
        yawn_count_after = len(self.yawn_times)
        multiple_yawns = yawn_count_after >= self.max_yawns_before_alert
        
        # Crear resultado completo
        result = {
            'detection_result': detection_result,
            'operator_info': self.current_operator,
            'operator_name': self.current_operator.get('name', 'Unknown'),
            'is_calibrated': self.is_calibrated,
            'calibration_confidence': self.current_thresholds.get('calibration_confidence', 0),
            'yawn_count': yawn_count_after,
            'window_minutes': self.window_size // 60,
            'max_yawns': self.max_yawns_before_alert,
            'multiple_yawns': multiple_yawns,
            'duration_threshold': self.current_thresholds.get('duration_threshold', 2.5),
            'timestamp': time.time()
        }
        
        # Dibujar información en el frame (si no se hizo antes)
        if not detection_result.get('is_yawning', False):
            frame_with_info = self.detector.draw_yawn_info(frame, detection_result)
        else:
            frame_with_info = frame_with_drawings
        
        # Agregar dashboard si está habilitado
        if self.dashboard_enabled:
            frame_with_info = self.dashboard.render(frame_with_info, result)
        
        result['frame'] = frame_with_info
        
        # Manejar evento de múltiples bostezos
        if multiple_yawns:
            self._handle_multiple_yawns(result)
        
        return result
    
    def _select_best_yawn_frame(self):
        """
        Selecciona el mejor frame del bostezo para capturar.
        Prioriza el frame con MAR máximo.
        """
        if not self.current_yawn_frames:
            return None
        
        # Estrategia principal: Frame con MAR máximo
        if self.frame_at_max_mar:
            self.logger.debug(f"Usando frame con MAR máximo: {self.frame_at_max_mar['mar']:.3f}")
            return self.frame_at_max_mar
        
        # Estrategia alternativa: Frame en el punto medio temporal
        middle_index = len(self.current_yawn_frames) // 2
        middle_frame = self.current_yawn_frames[middle_index]
        self.logger.debug(f"Usando frame medio (índice {middle_index})")
        return middle_frame
    
    def _reset_yawn_buffers(self):
        """Limpia los buffers de captura para el próximo bostezo"""
        self.max_mar_during_yawn = 0
        self.frame_at_max_mar = None
        self.current_yawn_frames.clear()
        self.yawn_start_time = None
    
    def _handle_yawn_detected(self, detection_result, capture_frame, capture_info=None):
        """
        Maneja la detección de un bostezo completo con frame específico.
        
        Args:
            detection_result: Resultado de la detección
            capture_frame: Frame capturado en el punto óptimo
            capture_info: Información sobre la captura
        """
        current_time = time.time()
        
        # Registrar bostezo
        self.yawn_times.append(current_time)
        self.session_stats['total_yawns'] += 1
        
        # Guardar datos del último bostezo
        self.last_yawn_data = {
            'frame': capture_frame,
            'capture_info': capture_info,
            'detection_result': detection_result,
            'timestamp': current_time
        }
        
        # Log detallado
        duration = detection_result.get('yawn_duration', 0)
        yawn_count = len(self.yawn_times)
        self.logger.info(f"Bostezo #{yawn_count} confirmado - Duración: {duration:.1f}s")
        
        # Reproducir audio según el número de bostezos
        if self.alarm_module and (current_time - self.last_audio_time > self.audio_cooldown):
            # Determinar qué audio reproducir
            if yawn_count == 1:
                audio_key = "bostezo1"
            elif yawn_count == 2:
                audio_key = "bostezo2"
            else:  # yawn_count >= 3
                audio_key = "bostezo3"
            
            self.logger.info(f"Reproduciendo: {audio_key}")
            success = self.alarm_module.play_audio(audio_key)
            
            if success:
                self.last_audio_time = current_time
    
    def _handle_multiple_yawns(self, result):
        """
        Maneja el evento de múltiples bostezos.
        Usa el frame capturado del último bostezo.
        """
        current_time = time.time()
        
        # Verificar cooldown de reporte
        if current_time - self.last_report_time < self.report_cooldown:
            return
        
        # Log de alerta
        self.logger.warning(
            f"ALERTA: {len(self.yawn_times)} bostezos en {self.window_size//60} minutos - "
            f"Operador: {self.current_operator['name']}"
        )
        
        # Incrementar contador de eventos
        self.session_stats['multiple_yawn_events'] += 1
        
        # Preparar datos del evento
        event_data = {
            'yawn_count': len(self.yawn_times),
            'window_minutes': self.window_size // 60,
            'max_allowed': self.max_yawns_before_alert,
            'yawn_times': list(self.yawn_times),
            'detection_details': result.get('detection_result', {}),
            'is_calibrated': self.is_calibrated,
            'analysis_timestamp': current_time
        }
        
        # Agregar información de captura si está disponible
        if self.last_yawn_data and self.last_yawn_data.get('capture_info'):
            event_data['capture_info'] = self.last_yawn_data['capture_info']
        
        # Usar el frame del último bostezo capturado si está disponible
        frame_to_save = result['frame']  # Frame actual con dashboard
        if self.last_yawn_data and 'frame' in self.last_yawn_data:
            # Aplicar dashboard al frame capturado
            frame_with_yawn_drawings = self.last_yawn_data['frame']

            # podemos dibujar texto adicional aquí
            cv2.putText(frame_with_yawn_drawings, 
                        f"Bostezo capturado - MAR: {self.last_yawn_data['capture_info']['capture_mar']:.3f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 255), 
                        2)
            
            # Aplicar dashboard
            frame_to_save = self.dashboard.render(frame_with_yawn_drawings, result)
        
        # Generar reporte
        report = self.report_manager.generate_report(
            module_name='yawn',
            event_type='multiple_yawns',
            data=event_data,
            frame=frame_to_save,
            operator_info=self.current_operator
        )
        
        if report:
            self.last_report_time = current_time
            self.session_stats['reports_generated'] += 1
            self.logger.info(f"Reporte de múltiples bostezos generado: {report['id']}")
    
    def _clean_old_yawns(self):
        """Elimina bostezos fuera de la ventana temporal"""
        current_time = time.time()
        
        # Eliminar bostezos antiguos
        while self.yawn_times and (current_time - self.yawn_times[0] > self.window_size):
            removed_time = self.yawn_times.popleft()
            self.logger.debug(f"Bostezo antiguo eliminado (más de {self.window_size//60} minutos)")
    
    def force_yawn_report(self, frame):
        """
        Fuerza el envío de un reporte de múltiples bostezos (para testing).
        """
        if len(self.yawn_times) > 0:
            # Crear resultado artificial
            result = {
                'yawn_count': len(self.yawn_times),
                'window_minutes': self.window_size // 60,
                'max_yawns': self.max_yawns_before_alert,
                'multiple_yawns': True,
                'operator_info': self.current_operator,
                'frame': frame
            }
            
            self._handle_multiple_yawns(result)
            self.logger.warning("REPORTE FORZADO: Múltiples bostezos")
            return True
        else:
            self.logger.info("No hay bostezos para reportar")
            return False
    
    def reset_yawn_counter(self):
        """Reinicia el contador de bostezos"""
        self.yawn_times.clear()
        self._reset_yawn_buffers()
        self.logger.info("Contador de bostezos reiniciado")
        return True
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        return {
            'current_operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'detector_status': self.detector.get_status(),
            'yawn_count': len(self.yawn_times),
            'yawn_times': list(self.yawn_times),
            'session_stats': self.session_stats,
            'capture_buffer_size': len(self.current_yawn_frames)
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
        self.yawn_times.clear()
        self.last_report_time = 0
        self.last_audio_time = 0
        self._reset_yawn_buffers()
        
        # Reiniciar estadísticas
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_yawns': 0,
            'multiple_yawn_events': 0,
            'operators_monitored': set(),
            'reports_generated': 0
        }
        
        # Reiniciar componentes
        self.detector.reset()
        self.dashboard.reset()
        
        self.logger.info("Sistema de detección de bostezos reiniciado")