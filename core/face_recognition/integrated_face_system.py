"""
Sistema Integrado de Reconocimiento Facial (SIMPLIFICADO)
=========================================================
Solo reconocimiento facial y detección de operadores no autorizados.
"""

import time
import logging
import os
from .face_recognition_module import FaceRecognitionModule
from .face_recognition_calibration import FaceRecognitionCalibration
from .face_recognition_dashboard import FaceRecognitionDashboard
from core.reports.report_manager import get_report_manager

class IntegratedFaceSystem:
    def __init__(self, operators_dir="operators", dashboard_position='right'):
        """
        Inicializa el sistema integrado de reconocimiento facial.
        
        Args:
            operators_dir: Directorio base de operadores
            dashboard_position: Posición del dashboard ('left' o 'right')
        """
        self.operators_dir = operators_dir
        self.logger = logging.getLogger('IntegratedFaceSystem')
        
        # Componentes del sistema
        self.calibration_manager = FaceRecognitionCalibration(
            baseline_dir=os.path.join(operators_dir, "baseline-json")
        )
        
        # Módulo de reconocimiento
        self.recognizer = FaceRecognitionModule(operators_dir)
        
        # Dashboard visual
        self.dashboard = FaceRecognitionDashboard(position=dashboard_position)
        self.dashboard_enabled = True
        
        # Gestor de reportes
        self.report_manager = get_report_manager()
        
        # Estado actual
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        
        # Control de operador desconocido
        self.unknown_operator_start_time = None
        self.unknown_operator_reported = False
        self.unknown_operator_threshold = 15 * 60  # 15 minutos en segundos
        
        # Control de audio para operador desconocido con intervalos
        self.unknown_audio_intervals = [0, 5*60, 10*60, 15*60]  # 0, 5, 10 y 15 minutos
        self.unknown_audio_played_at = []  # Tiempos en que se reprodujo el audio
        
        # Control para el audio inicial
        self.ultimo_operador_id = None
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'unknown_detections': 0,
            'operators_detected': set(),
            'unknown_operator_reports': 0
        }
        
        # Cargar operadores
        if not self.recognizer.load_operators():
            self.logger.error("No se pudieron cargar los operadores")
    
    def identify_and_analyze(self, frame):
        """
        Identifica al operador en el frame.
        
        Args:
            frame: Frame de video
            
        Returns:
            dict: Resultados del análisis
        """
        # Incrementar contador
        self.session_stats['total_recognitions'] += 1
        
        # Realizar identificación
        operator_info = self.recognizer.identify_operator(frame)
        
        # Crear resultado estructurado
        result = {
            'operator_info': operator_info,
            'timestamp': time.time(),
            'is_calibrated': False,
            'thresholds_used': None
        }
        
        if operator_info:
            if operator_info.get('is_registered', False):
                # OPERADOR REGISTRADO
                self.session_stats['successful_recognitions'] += 1
                operator_id = operator_info['id']
                
                # Resetear contador de desconocido
                self._reset_unknown_operator()
                
                # Si es un operador diferente al actual
                if not self.current_operator or self.current_operator['id'] != operator_id:
                    self._handle_operator_change(operator_info)
                
                # Actualizar resultado con calibración
                result['is_calibrated'] = self.is_calibrated
                result['thresholds_used'] = self.current_thresholds
                
            else:
                # OPERADOR NO REGISTRADO
                self.session_stats['unknown_detections'] += 1
                self._handle_unknown_operator(frame)
                
                # Limpiar operador actual
                if self.current_operator:
                    self.logger.info(f"Operador {self.current_operator['name']} ya no está presente")
                    self.current_operator = None
                    self.is_calibrated = False
        else:
            # NO HAY ROSTRO DETECTADO
            self._reset_unknown_operator()
            
            if self.current_operator:
                self.logger.info(f"Operador {self.current_operator['name']} ya no está presente")
                self.current_operator = None
                self.is_calibrated = False
        
        # Dibujar información en el frame
        frame_with_info = self.recognizer.draw_operator_info(frame, operator_info)
        
        # Agregar dashboard si está habilitado
        if self.dashboard_enabled:
            frame_with_info = self.dashboard.render(frame_with_info, result)
        
        result['frame'] = frame_with_info
        
        return result
    
    def _handle_operator_change(self, new_operator):
        """
        Maneja el cambio de operador registrado.
        
        Args:
            new_operator: Nueva información del operador
        """
        operator_id = new_operator['id']
        self.logger.info(f"Operador detectado: {new_operator['name']} ({operator_id})")
        
        # Establecer nuevo operador
        self.current_operator = new_operator
        self.session_stats['operators_detected'].add(operator_id)
        
        # Cargar calibración personalizada
        thresholds = self.calibration_manager.get_thresholds(operator_id)
        
        if thresholds.get('calibration_confidence', 0) > 0:
            self.is_calibrated = True
            self.logger.info(f"Calibración personalizada cargada para {operator_id}")
        else:
            self.is_calibrated = False
            self.logger.warning(f"Usando valores por defecto para {operator_id}")
        
        # Actualizar módulo con nuevos umbrales
        self.current_thresholds = thresholds
        self.recognizer.update_config(thresholds)
    
    def _handle_unknown_operator(self, frame, operator_info=None):
        """
        Maneja la detección de operador no registrado con audio en intervalos.
        
        Args:
            frame: Frame actual
            operator_info: Información del operador desconocido (opcional)
        """
        current_time = time.time()
        
        if self.unknown_operator_start_time is None:
            # Primera detección de operador desconocido
            self.unknown_operator_start_time = current_time
            self.unknown_operator_reported = False
            self.unknown_audio_played_at = [0]  # Marcar que ya se reprodujo el audio inicial
            self.logger.warning("Operador no registrado detectado")
        
        # Calcular tiempo transcurrido
        elapsed_time = current_time - self.unknown_operator_start_time
        
        # Verificar si debemos reproducir audio según intervalos (5, 10, 15 min)
        for interval in self.unknown_audio_intervals[1:]:  # Saltar el primer intervalo (0)
            # Si hemos alcanzado este intervalo y no hemos reproducido audio en él
            if elapsed_time >= interval and interval not in self.unknown_audio_played_at:
                # Reproducir audio
                self.recognizer.reproducir_audio("assets/audio/no_registrado.mp3")
                self.unknown_audio_played_at.append(interval)
                
                # Log del evento
                minutes = int(interval / 60)
                self.logger.warning(f"Audio: Operador no registrado por {minutes} minutos")
        
        # Generar reporte a los 15 minutos
        if elapsed_time >= self.unknown_operator_threshold and not self.unknown_operator_reported:
            self._generate_unknown_operator_report(frame, elapsed_time)
            self.unknown_operator_reported = True
            self.session_stats['unknown_operator_reports'] += 1
    
    def force_unknown_operator_report(self, frame):
        """
        Fuerza el envío de un reporte de operador desconocido (para testing).
        
        Args:
            frame: Frame actual
            
        Returns:
            bool: True si se generó el reporte
        """
        if self.unknown_operator_start_time is not None:
            elapsed_time = time.time() - self.unknown_operator_start_time
            self._generate_unknown_operator_report(frame, elapsed_time)
            self.unknown_operator_reported = True
            self.session_stats['unknown_operator_reports'] += 1
            self.logger.warning("REPORTE FORZADO: Operador desconocido")
            return True
        else:
            self.logger.info("No hay operador desconocido activo para reportar")
            return False
    
    def _reset_unknown_operator(self):
        """Resetea el contador de operador desconocido"""
        if self.unknown_operator_start_time is not None:
            elapsed = time.time() - self.unknown_operator_start_time
            if elapsed > 10:  # Solo log si estuvo más de 10 segundos
                self.logger.info(f"Operador desconocido estuvo presente por {elapsed:.0f} segundos")
        
        self.unknown_operator_start_time = None
        self.unknown_operator_reported = False
        self.unknown_audio_played_at = []  # Resetear audio también
        self.ultimo_operador_id = None  # Resetear control de audio
    
    def _generate_unknown_operator_report(self, frame, elapsed_time):
        """
        Genera reporte de operador desconocido.
        
        Args:
            frame: Frame actual
            elapsed_time: Tiempo que lleva el operador desconocido
        """
        try:
            event_data = {
                'event_type': 'unknown_operator',
                'duration_seconds': elapsed_time,
                'duration_minutes': elapsed_time / 60,
                'threshold_minutes': self.unknown_operator_threshold / 60,
                'timestamp': time.time(),
                'message': f"Operador no registrado detectado por {elapsed_time/60:.1f} minutos"
            }
            
            report = self.report_manager.generate_report(
                module_name='face_recognition',
                event_type='unknown_operator_15min',
                data=event_data,
                frame=frame,
                operator_info={'id': 'UNKNOWN', 'name': 'No Registrado'}
            )
            
            if report:
                self.logger.warning(f"ALERTA: Reporte de operador desconocido generado: {report['id']}")
                
        except Exception as e:
            self.logger.error(f"Error generando reporte de operador desconocido: {e}")
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        recognizer_status = self.recognizer.get_status()
        
        # Calcular tiempo de operador desconocido si aplica
        unknown_time = 0
        if self.unknown_operator_start_time:
            unknown_time = time.time() - self.unknown_operator_start_time
        
        return {
            'current_operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'recognizer_status': recognizer_status,
            'session_stats': self.session_stats,
            'unknown_operator_time': unknown_time,
            'unknown_operator_active': self.unknown_operator_start_time is not None
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
        self.unknown_operator_start_time = None
        self.unknown_operator_reported = False
        
        # Reiniciar estadísticas
        self.session_stats = {
            'session_start': time.time(),
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'unknown_detections': 0,
            'operators_detected': set(),
            'unknown_operator_reports': 0
        }
        
        # Reiniciar dashboard
        if hasattr(self, 'dashboard'):
            self.dashboard.reset()
        
        self.logger.info("Sistema de reconocimiento facial reiniciado")