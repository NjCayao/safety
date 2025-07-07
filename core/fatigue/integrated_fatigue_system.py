"""
Sistema Integrado de Detección de Fatiga
========================================
Orquesta la detección de fatiga con calibración personalizada.
"""

import time
import logging
import os
from core.reports.report_manager import get_report_manager
from .fatigue_detection import FatigueDetector
from .fatigue_calibration import FatigueCalibration

class IntegratedFatigueSystem:
    def __init__(self, operators_dir="operators", model_path="assets/models/shape_predictor_68_face_landmarks.dat", headless=False):
        """
        Inicializa el sistema integrado de fatiga.
        
        Args:
            operators_dir: Directorio base de operadores
            model_path: Ruta al modelo de landmarks
            headless: Si True, modo sin GUI
        """
        self.report_manager = get_report_manager()
        self.operators_dir = operators_dir
        self.model_path = model_path
        self.headless = headless
        self.logger = logging.getLogger('IntegratedFatigueSystem')

        # Configuración de reportes
        self.report_config = {
            'send_on_microsleep': True,
            'send_on_critical': True,
            'include_frame': True,
            'cooldown_seconds': 30  # Evitar spam de reportes
        }
        # Control de reportes
        self.last_report_time = {}
        
        # Componentes del sistema
        self.calibration_manager = FatigueCalibration(
            baseline_dir=os.path.join(operators_dir, "baseline-json")
        )
        self.detector = None
        
        # Estado actual
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        
        # Historial para análisis
        self.detection_history = []
        self.max_history_size = 300  # ~10 segundos a 30fps
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_microsleeps': 0,
            'total_alerts': 0,
            'operators_monitored': set()
        }
        
        # Inicializar componentes
        self._initialize_components()
        
    def _initialize_components(self):
        """Inicializa los componentes del sistema"""
        # Inicializar calibrador
        if not self.calibration_manager.initialize_detectors(self.model_path):
            self.logger.error("Error inicializando detectores de calibración")
        
        # Crear detector con valores por defecto
        self.detector = FatigueDetector(self.model_path, headless=self.headless)
        
        self.logger.info("Sistema integrado de fatiga inicializado")
    
    def set_operator(self, operator_info):
        """
        Establece el operador actual y carga su calibración.
        
        Args:
            operator_info: Dict con 'id' y 'name' del operador
            
        Returns:
            bool: True si se cargó correctamente
        """
        if not operator_info or 'id' not in operator_info:
            self.logger.error("Información de operador inválida")
            return False
        
        operator_id = operator_info['id']
        
        # Si es el mismo operador, no hacer nada
        if self.current_operator and self.current_operator['id'] == operator_id:
            return True
        
        self.logger.info(f"Cambiando a operador: {operator_info.get('name', 'Unknown')} ({operator_id})")
        
        # Guardar estadísticas del operador anterior
        if self.current_operator:
            self._save_operator_session()
        
        # Establecer nuevo operador
        self.current_operator = operator_info
        self.session_stats['operators_monitored'].add(operator_id)
        
        # Cargar calibración personalizada
        thresholds = self.calibration_manager.get_thresholds(operator_id)
        
        if thresholds.get('calibration_confidence', 0) > 0:
            self.is_calibrated = True
            self.logger.info(f"Calibración personalizada cargada para {operator_id}")
        else:
            self.is_calibrated = False
            self.logger.warning(f"Usando valores por defecto para {operator_id}")
        
        # Actualizar detector con nuevos umbrales
        self.current_thresholds = thresholds
        self.detector.update_thresholds(thresholds)
        
        # Resetear detector para nuevo operador
        self.detector.reset()
        self.detection_history.clear()
        
        return True

    def analyze_frame(self, frame, face_landmarks):
        """
        Analiza un frame para detectar fatiga.
        
        Args:
            frame: Frame de video
            face_landmarks: Landmarks faciales detectados
            
        Returns:
            dict: Resultados del análisis
        """
        if not self.current_operator:
            return {
                'status': 'error',
                'message': 'No hay operador activo',
                'timestamp': time.time()
            }
        
        # Incrementar contador
        self.session_stats['total_detections'] += 1
        
        # El detector original retorna: (microsleep_detected, critical_fatigue, analyzed_frame)
        microsleep_detected, critical_fatigue, analyzed_frame = self.detector.detect(frame)
        
        # Crear resultado estructurado
        result = {
            'ear_value': self.detector.last_ear_values[-1] if self.detector.last_ear_values else 0.0,
            'smooth_ear': sum(self.detector.last_ear_values) / len(self.detector.last_ear_values) if self.detector.last_ear_values else 0.0,
            'ear_threshold': self.detector.EAR_THRESHOLD,
            'eyes_closed': self.detector.eyes_closed_duration > 0,
            'eyes_closed_duration': self.detector.eyes_closed_duration,
            'microsleep_detected': microsleep_detected,
            'microsleep_count': len(self.detector.microsleeps),
            'total_microsleeps': len(self.detector.microsleeps),
            'blink_rate': 0,  # El detector original no trackea blinks
            'fatigue_percentage': self._calculate_fatigue_percentage(),
            'is_critical': critical_fatigue,
            'is_night_mode': self.detector.is_night_mode,
            'light_level': self.detector.light_level,
            'timestamp': time.time(),
            'operator_id': self.current_operator['id'],
            'operator_name': self.current_operator.get('name', 'Unknown'),
            'is_calibrated': self.is_calibrated,
            'frame': analyzed_frame  # Incluir el frame analizado
        }
        
        # Procesar eventos significativos
        if result.get('microsleep_detected', False):
            self.session_stats['total_microsleeps'] += 1
            # Solo manejar el evento si tenemos el método (por ahora comentado)
            if hasattr(self, '_handle_microsleep_event'):
                self._handle_microsleep_event(result, frame)
        
        # Verificar fatiga crítica
        if result.get('is_critical', False):
            # Solo manejar si tenemos el método (por ahora comentado)
            if hasattr(self, '_handle_critical_fatigue'):
                self._handle_critical_fatigue(result, frame)
        
        # Agregar al historial
        self._update_history(result)
        
        # Generar alertas si es necesario
        alerts = self._check_alerts(result)
        if alerts:
            result['alerts'] = alerts
            self.session_stats['total_alerts'] += len(alerts)
        
        # Calcular tendencias
        result['trends'] = self._calculate_trends()
        
        return result
    
    def _handle_microsleep_event(self, result, frame=None):
        """Maneja un evento de microsueño con generación de reporte"""
        self.logger.warning(
            f"Microsueño detectado - Operador: {result['operator_name']}, "
            f"Duración: {result['eyes_closed_duration']:.1f}s"
        )
        
        # Verificar cooldown
        current_time = time.time()
        last_time = self.last_report_time.get('microsleep', 0)
        
        if current_time - last_time < self.report_config['cooldown_seconds']:
            return
        
        # Generar reporte si está habilitado
        if self.report_config['send_on_microsleep']:
            # Preparar datos del evento
            event_data = {
                'ear_value': result.get('ear_value', 0),
                'ear_threshold': result.get('ear_threshold', 0.25),
                'eyes_closed_duration': result.get('eyes_closed_duration', 0),
                'microsleep_count': result.get('microsleep_count', 0),
                'is_night_mode': result.get('is_night_mode', False),
                'light_level': result.get('light_level', 0),
                'fatigue_percentage': result.get('fatigue_percentage', 0),
                'analysis_timestamp': result.get('timestamp', time.time())
            }
            
            # IMPORTANTE: Usar el frame procesado que incluye dashboard
            # El frame con dashboard está en result['frame']
            frame_to_save = result.get('frame', frame)
            
            # Generar reporte
            report = self.report_manager.generate_report(
                module_name='fatigue',
                event_type='microsleep',
                data=event_data,
                frame=frame_to_save if self.report_config['include_frame'] else None,
                operator_info=self.current_operator
            )
            
            if report:
                self.last_report_time['microsleep'] = current_time
                self.logger.info(f"Reporte de microsueño generado: {report['id']}")

    def _handle_critical_fatigue(self, result, frame=None):
        """Maneja evento de fatiga crítica con reporte"""
        # Verificar cooldown
        current_time = time.time()
        last_time = self.last_report_time.get('critical', 0)
        
        if current_time - last_time < self.report_config['cooldown_seconds'] * 2:  # Mayor cooldown para críticos
            return
        
        if self.report_config['send_on_critical']:
            # Preparar datos del evento crítico
            event_data = {
                'microsleep_count': result.get('microsleep_count', 0),
                'total_microsleeps': result.get('total_microsleeps', 0),
                'fatigue_percentage': result.get('fatigue_percentage', 0),
                'alerts': result.get('alerts', []),
                'session_duration': time.time() - self.session_stats['session_start'],
                'critical_timestamp': time.time()
            }
            
            # Generar reporte crítico
            report = self.report_manager.generate_report(
                module_name='fatigue',
                event_type='critical_fatigue',
                data=event_data,
                frame=frame if self.report_config['include_frame'] else None,
                operator_info=self.current_operator
            )
            
            if report:
                self.last_report_time['critical'] = current_time
                self.logger.critical(f"Reporte crítico generado: {report['id']}")
    
    def _check_alerts(self, result):
        """Verifica si se deben generar alertas"""
        alerts = []
        
        # Alerta por fatiga crítica
        if result.get('is_critical', False):
            alerts.append({
                'type': 'critical_fatigue',
                'severity': 'high',
                'message': 'Fatiga crítica detectada',
                'timestamp': time.time()
            })
        
        # Alerta por microsueños múltiples
        if result.get('microsleep_count', 0) >= 3:
            alerts.append({
                'type': 'multiple_microsleeps',
                'severity': 'critical',
                'message': f"{result['microsleep_count']} microsueños en 10 minutos",
                'timestamp': time.time()
            })
        
        # Alerta por duración prolongada
        if result.get('eyes_closed_duration', 0) > 3.0:
            alerts.append({
                'type': 'prolonged_closure',
                'severity': 'critical',
                'message': f"Ojos cerrados por {result['eyes_closed_duration']:.1f}s",
                'timestamp': time.time()
            })
        
        return alerts
    
    def _update_history(self, result):
        """Actualiza el historial de detecciones"""
        # Agregar solo datos relevantes para tendencias
        history_entry = {
            'timestamp': result['timestamp'],
            'ear_value': result['ear_value'],
            'fatigue_percentage': result['fatigue_percentage'],
            'microsleep_detected': result['microsleep_detected'],
            'eyes_closed': result['eyes_closed']
        }
        
        self.detection_history.append(history_entry)
        
        # Limitar tamaño del historial
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def _calculate_trends(self):
        """Calcula tendencias basadas en el historial"""
        if len(self.detection_history) < 10:
            return None
        
        # Análisis de los últimos 30 segundos vs los primeros 30 segundos
        recent_data = self.detection_history[-90:] if len(self.detection_history) > 90 else self.detection_history
        
        # Calcular promedios
        recent_fatigue = sum(d['fatigue_percentage'] for d in recent_data) / len(recent_data)
        recent_ear = sum(d['ear_value'] for d in recent_data) / len(recent_data)
        
        # Comparar con datos anteriores si hay suficiente historial
        trend = 'stable'
        if len(self.detection_history) > 180:
            older_data = self.detection_history[:90]
            older_fatigue = sum(d['fatigue_percentage'] for d in older_data) / len(older_data)
            
            if recent_fatigue > older_fatigue + 10:
                trend = 'increasing'
            elif recent_fatigue < older_fatigue - 10:
                trend = 'decreasing'
        
        return {
            'fatigue_trend': trend,
            'current_avg_fatigue': recent_fatigue,
            'current_avg_ear': recent_ear,
            'data_points': len(self.detection_history)
        }
    
    def force_calibration(self, operator_id, photos_path):
        """
        Fuerza una calibración para un operador específico.
        
        Args:
            operator_id: ID del operador
            photos_path: Ruta a las fotos del operador
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Forzando calibración para operador {operator_id}")
        
        calibration = self.calibration_manager.calibrate_from_photos(operator_id, photos_path)
        
        if calibration:
            # Si es el operador actual, actualizar umbrales
            if self.current_operator and self.current_operator['id'] == operator_id:
                self.current_thresholds = calibration['thresholds']
                self.detector.update_thresholds(calibration['thresholds'])
                self.is_calibrated = True
                self.logger.info("Umbrales actualizados con nueva calibración")
            
            return True
        
        return False
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        detector_stats = self.detector.get_statistics() if self.detector else {}
        
        return {
            'operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'detector_stats': detector_stats,
            'session_stats': self.session_stats,
            'history_size': len(self.detection_history)
        }
    
    def get_report(self):
        """Genera un reporte completo para el servidor"""
        current_time = time.time()
        session_duration = current_time - self.session_stats['session_start']
        
        report = {
            'module': 'fatigue',
            'version': '1.0',
            'timestamp': current_time,
            'session': {
                'duration_seconds': session_duration,
                'operators_monitored': list(self.session_stats['operators_monitored']),
                'total_detections': self.session_stats['total_detections'],
                'total_microsleeps': self.session_stats['total_microsleeps'],
                'total_alerts': self.session_stats['total_alerts']
            },
            'current_operator': self.current_operator,
            'current_status': self.detector.get_statistics() if self.detector else None,
            'calibration': {
                'is_calibrated': self.is_calibrated,
                'confidence': self.current_thresholds.get('calibration_confidence', 0) if self.current_thresholds else 0
            }
        }
        
        # Agregar últimas detecciones significativas
        recent_microsleeps = [
            d for d in self.detection_history[-30:] 
            if d.get('microsleep_detected', False)
        ]
        
        if recent_microsleeps:
            report['recent_events'] = recent_microsleeps
        
        return report
    
    def _save_operator_session(self):
        """Guarda estadísticas de la sesión del operador"""
        # Aquí se podría guardar en archivo o base de datos
        # Por ahora solo log
        if self.current_operator:
            self.logger.info(
                f"Sesión finalizada para {self.current_operator.get('name', 'Unknown')} - "
                f"Microsueños: {self.detector.total_microsleeps}"
            )
    
    def reset(self):
        """Reinicia el sistema completamente"""
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        self.detection_history.clear()
        
        if self.detector:
            self.detector.reset()
        
        # Reiniciar estadísticas
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_microsleeps': 0,
            'total_alerts': 0,
            'operators_monitored': set()
        }
        
        self.logger.info("Sistema reiniciado")

    def _calculate_fatigue_percentage(self):
        """Calcula porcentaje de fatiga basado en microsueños"""
        if not self.detector:
            return 0
        
        microsleeps = len(self.detector.microsleeps)
        
        if microsleeps >= 3:
            return 100
        elif microsleeps == 2:
            return 70
        elif microsleeps == 1:
            return 40
        else:
            return 20 if self.detector.eyes_closed_duration > 0 else 0
        
    def generate_session_report(self):
        """Genera reporte de sesión completa"""
        try:
            session_data = {
                'session_start': self.session_stats['session_start'],
                'session_end': time.time(),
                'duration_seconds': time.time() - self.session_stats['session_start'],
                'operators_monitored': list(self.session_stats['operators_monitored']),
                'total_detections': self.session_stats['total_detections'],
                'total_microsleeps': self.session_stats['total_microsleeps'],
                'total_alerts': self.session_stats['total_alerts'],
                'detector_stats': self.detector.get_statistics() if self.detector else None,
                'calibration_used': self.is_calibrated
            }
            
            # Generar reporte de sesión
            report = self.report_manager.generate_report(
                module_name='fatigue',
                event_type='session_summary',
                data=session_data,
                operator_info=self.current_operator
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de sesión: {e}")
            return None
    
    def update_report_config(self, new_config):
        """Actualiza configuración de reportes"""
        self.report_config.update(new_config)
        self.logger.info("Configuración de reportes actualizada")