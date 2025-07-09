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
        
        # Crear e inicializar AlarmModule
        self.alarm_module = AlarmModule(audio_dir="assets/audio")
        self.alarm_module.initialize()
        
        # Crear detector y conectar AlarmModule
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
        self.first_distraction_time = None  # Para rastrear el inicio del ciclo
        
        # Control de reportes
        self.last_report_time = 0
        self.report_cooldown = 30  # segundos
        
        # Control de audio
        self.last_audio_time = 0
        self.audio_cooldown = 5  # segundos
        
        # Variables para captura mejorada
        self.current_distraction_frames = []
        self.max_rotation_during_distraction = 0
        self.frame_at_max_rotation = None
        self.distraction_start_time = None
        self.last_distraction_data = None
        self.extreme_event_recorded = False
        
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
        
        self.logger.info("Sistema integrado de distracciones inicializado")
    
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
        self.max_distractions_before_alert = 3
        
        # Limpiar historiales
        self._reset_distraction_cycle()
        
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
        
        # Verificar si el ciclo actual ha expirado (10 minutos)
        self._check_cycle_expiration()
        
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
                
                # Si es el primer evento del ciclo, marcar inicio
                if len(self.distraction_times) == 1:
                    self.first_distraction_time = current_time
                
                # Capturar frame en este momento
                self.last_distraction_data = {
                    'frame': frame.copy(),
                    'timestamp': current_time,
                    'duration': duration
                }
                
                # Log
                count = len(self.distraction_times)
                self.logger.warning(f"⚠️ GIRO EXTREMO #{count} registrado (duración: {duration:.1f}s)")
        else:
            # Volvió a posición normal
            if self.distraction_start_time is not None:
                duration = time.time() - self.distraction_start_time
                if duration >= 3.0:
                    self.logger.info(f"✅ Volvió a posición normal después de {duration:.1f}s")
            
            # Resetear
            self.distraction_start_time = None
            self.extreme_event_recorded = False
        
        # Verificar si hay 3 giros extremos
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
            'calibration_confidence': self.current_thresholds.get('calibration_confidence', 0),
            'total_distractions': total_extreme_events,
            'window_minutes': self.window_size // 60,
            'max_distractions': 3,
            'timestamp': time.time()
        }
        
        # Aplicar dashboard ANTES de guardar para captura
        if self.dashboard_enabled:
            frame_with_dashboard = self.dashboard.render(frame.copy(), result)
        else:
            frame_with_dashboard = frame.copy()
        
        result['frame'] = frame_with_dashboard
        
        # Generar reporte si hay 3 giros extremos
        if should_report and not self.report_recently_sent():
            self._generate_extreme_rotation_report(result)
            # IMPORTANTE: Reiniciar ciclo después del reporte
            self._reset_distraction_cycle()
            self.logger.info("🔄 Ciclo reiniciado después del reporte (3/3)")
        
        return result
    
    def _check_cycle_expiration(self):
        """Verifica si han pasado 10 minutos desde el primer evento"""
        if self.first_distraction_time and len(self.distraction_times) > 0:
            current_time = time.time()
            elapsed = current_time - self.first_distraction_time
            
            if elapsed > self.window_size:
                # El ciclo ha expirado
                count = len(self.distraction_times)
                self.logger.info(f"⏰ Ciclo expirado después de {self.window_size//60} minutos con {count}/3 eventos")
                self._reset_distraction_cycle()
    
    def _reset_distraction_cycle(self):
        """Reinicia el ciclo de conteo de distracciones"""
        self.distraction_times.clear()
        self.first_distraction_time = None
        self.current_distraction_frames.clear()
        self.last_distraction_data = None
        self.logger.info("🔄 Contador de distracciones reiniciado (0/3)")
    
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
        
        # Calcular tiempo del ciclo
        if self.first_distraction_time:
            cycle_duration = current_time - self.first_distraction_time
            cycle_minutes = cycle_duration / 60
        else:
            cycle_minutes = 0
        
        # Preparar datos
        event_data = {
            'extreme_rotation_count': len(self.distraction_times),
            'window_minutes': self.window_size // 60,
            'actual_cycle_minutes': round(cycle_minutes, 1),
            'event_times': list(self.distraction_times),
            'operator_id': self.current_operator['id'],
            'operator_name': self.current_operator['name'],
            'severity': 'HIGH',
            'recommendation': 'Verificar estado del operador inmediatamente'
        }
        
        # Usar el frame CON dashboard que viene en result
        frame_to_save = result['frame']  # Este ya tiene el dashboard
        
        # Si hay un frame de distracción guardado, usarlo con dashboard
        if self.last_distraction_data and 'frame' in self.last_distraction_data:
            # Aplicar dashboard al frame de distracción
            temp_result = result.copy()
            frame_with_dashboard = self.dashboard.render(
                self.last_distraction_data['frame'].copy(), 
                temp_result
            )
            
            # Agregar texto de alerta
            cv2.putText(frame_with_dashboard, 
                        "ALERTA: 3 GIROS EXTREMOS DETECTADOS", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, 
                        (0, 0, 255), 
                        3)
            
            cv2.putText(frame_with_dashboard, 
                        f"Completados en {cycle_minutes:.1f} minutos", 
                        (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (0, 0, 255), 
                        2)
            
            frame_to_save = frame_with_dashboard
        
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
            self.session_stats['multiple_distraction_events'] += 1
            self.logger.info(f"📋 Reporte de giros extremos generado: {report['id']}")
    
    def report_recently_sent(self):
        """Verifica si se envió un reporte recientemente"""
        return (time.time() - self.last_report_time) < self.report_cooldown
    
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
            
            self._generate_extreme_rotation_report(result)
            self._reset_distraction_cycle()  # Reiniciar después del reporte forzado
            self.logger.warning("REPORTE FORZADO: Múltiples distracciones")
            return True
        else:
            self.logger.info("No hay distracciones para reportar")
            return False
    
    def reset_distraction_counter(self):
        """Reinicia el contador de distracciones manualmente"""
        count = len(self.distraction_times)
        self._reset_distraction_cycle()
        self.logger.info(f"Contador reiniciado manualmente (tenía {count} eventos)")
        return True
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        # Calcular tiempo del ciclo actual
        if self.first_distraction_time and len(self.distraction_times) > 0:
            cycle_elapsed = time.time() - self.first_distraction_time
            cycle_remaining = max(0, self.window_size - cycle_elapsed)
        else:
            cycle_elapsed = 0
            cycle_remaining = self.window_size
        
        return {
            'current_operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'detector_status': self.detector.get_status(),
            'distraction_count': len(self.distraction_times),
            'distraction_times': list(self.distraction_times),
            'cycle_elapsed_seconds': cycle_elapsed,
            'cycle_remaining_seconds': cycle_remaining,
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
        self.last_report_time = 0
        self.last_audio_time = 0
        self._reset_distraction_cycle()
        
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
        self.detector.set_alarm_module(self.alarm_module)
        self.dashboard.reset()
        
        self.logger.info("Sistema de detección de distracciones reiniciado completamente")