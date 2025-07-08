"""
Sistema Integrado de Detección de Bostezos
==========================================
Orquesta la detección de bostezos con calibración personalizada y reportes.
"""

import time
import logging
import os
import pygame
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
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_yawns': 0,
            'multiple_yawn_events': 0,
            'operators_monitored': set(),
            'reports_generated': 0
        }
        
        self.logger.info("Sistema integrado de bostezos inicializado")
    
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
        
        return True
    
    def analyze_frame(self, frame, landmarks):
        """
        Analiza un frame para detectar bostezos.
        
        Args:
            frame: Frame de video
            landmarks: Landmarks faciales
            
        Returns:
            dict: Resultados del análisis con frame procesado
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
        
        # IMPORTANTE: Guardar el estado ANTES de manejar el bostezo
        yawn_count_before = len(self.yawn_times)
        
        # Actualizar historial si se detectó un bostezo completo
        if detection_result.get('yawn_detected', False):
            self._handle_yawn_detected(detection_result)
        
        # Limpiar bostezos antiguos
        self._clean_old_yawns()
        
        # Verificar si AHORA tenemos múltiples bostezos (después de agregar el nuevo)
        yawn_count_after = len(self.yawn_times)
        multiple_yawns = yawn_count_after >= self.max_yawns_before_alert
        
        # DEBUG
        if yawn_count_before == 2 and yawn_count_after == 3:
            self.logger.info("=== TERCER BOSTEZO DETECTADO ===")
            self.logger.info("El audio bostezo3 ya debería haberse reproducido")
            self.logger.info("NO se debe reproducir alarma adicional")
        
        # Crear resultado completo
        result = {
            'detection_result': detection_result,
            'operator_info': self.current_operator,
            'operator_name': self.current_operator.get('name', 'Unknown'),
            'is_calibrated': self.is_calibrated,
            'calibration_confidence': self.current_thresholds.get('calibration_confidence', 0),
            'yawn_count': yawn_count_after,  # Usar el contador actualizado
            'window_minutes': self.window_size // 60,
            'max_yawns': self.max_yawns_before_alert,
            'multiple_yawns': multiple_yawns,
            'duration_threshold': self.current_thresholds.get('duration_threshold', 2.5),
            'timestamp': time.time()
        }
        
        # Dibujar información en el frame
        frame_with_info = self.detector.draw_yawn_info(frame, detection_result)
        
        # Agregar dashboard si está habilitado
        if self.dashboard_enabled:
            frame_with_info = self.dashboard.render(frame_with_info, result)
        
        # IMPORTANTE: Guardar el frame con dashboard
        result['frame'] = frame_with_info
        
        # Manejar evento de múltiples bostezos SOLO para el reporte
        if multiple_yawns:
            self._handle_multiple_yawns(result, frame_with_info)
        
        return result
    
    def _handle_yawn_detected(self, detection_result):
        """Maneja la detección de un bostezo completo"""
        current_time = time.time()
        
        # Registrar bostezo
        self.yawn_times.append(current_time)
        self.session_stats['total_yawns'] += 1
        
        # Log detallado
        duration = detection_result.get('yawn_duration', 0)
        yawn_count = len(self.yawn_times)
        self.logger.info(f"Bostezo detectado #{yawn_count} - Duración: {duration:.1f}s")
        
        # Reproducir audio según el número de bostezos
        if self.alarm_module and (current_time - self.last_audio_time > self.audio_cooldown):
            # Log específico para debugging
            self.logger.info(f"=== AUDIO: Bostezo #{yawn_count} detectado ===")
            
            # Determinar qué audio reproducir
            if yawn_count == 1:
                audio_key = "bostezo1"
            elif yawn_count == 2:
                audio_key = "bostezo2"
            else:  # yawn_count >= 3
                audio_key = "bostezo3"
                self.logger.info(f"TERCER BOSTEZO O MÁS (#{yawn_count}) - Usando bostezo3")
            
            self.logger.info(f"Reproduciendo: {audio_key}")
            success = self.alarm_module.play_audio(audio_key)
            
            if success:
                self.last_audio_time = current_time
                self.logger.info(f"✅ Audio {audio_key} reproducido exitosamente")
            else:
                self.logger.error(f"❌ Error al reproducir {audio_key}")
        else:
            if not self.alarm_module:
                self.logger.error("AlarmModule no está inicializado")
            else:
                remaining = self.audio_cooldown - (current_time - self.last_audio_time)
                self.logger.info(f"⏳ Cooldown activo: {remaining:.1f}s restantes")
    
    def _handle_multiple_yawns(self, result, frame):
        """Maneja el evento de múltiples bostezos"""
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
        
        # Generar reporte
        event_data = {
            'yawn_count': len(self.yawn_times),
            'window_minutes': self.window_size // 60,
            'max_allowed': self.max_yawns_before_alert,
            'yawn_times': list(self.yawn_times),
            'detection_details': result.get('detection_result', {}),
            'is_calibrated': self.is_calibrated,
            'analysis_timestamp': current_time
        }
        
        # Incluir el frame con dashboard
        frame_to_save = result.get('frame', frame)
        
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
            
            # Reproducir alerta especial
            if self.alarm_module:
                self.alarm_module.play_audio("alarma")
    
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
        
        Args:
            frame: Frame actual
            
        Returns:
            bool: True si se generó el reporte
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
            
            self._handle_multiple_yawns(result, frame)
            self.logger.warning("REPORTE FORZADO: Múltiples bostezos")
            return True
        else:
            self.logger.info("No hay bostezos para reportar")
            return False
    
    def reset_yawn_counter(self):
        """Reinicia el contador de bostezos"""
        self.yawn_times.clear()
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
            'session_stats': self.session_stats
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