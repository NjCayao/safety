"""
Sistema Integrado de Detección de Comportamientos
=================================================
Orquesta la detección de comportamientos con calibración personalizada.
"""

import time
import logging
import os
from .behavior_detection_module import BehaviorDetectionModule
from .behavior_calibration import BehaviorCalibration
from core.reports.report_manager import get_report_manager

class IntegratedBehaviorSystem:
    def __init__(self, model_dir="assets/models", audio_dir="assets/audio", operators_dir="operators"):
        """
        Inicializa el sistema integrado de comportamientos.
        
        Args:
            model_dir: Directorio de modelos YOLO
            audio_dir: Directorio de archivos de audio
            operators_dir: Directorio base de operadores
        """
        self.model_dir = model_dir
        self.audio_dir = audio_dir
        self.operators_dir = operators_dir
        self.logger = logging.getLogger('IntegratedBehaviorSystem')
        
        # Componentes del sistema
        self.calibration_manager = BehaviorCalibration(
            baseline_dir=os.path.join(operators_dir, "baseline-json")
        )
        self.detector = BehaviorDetectionModule(model_dir, audio_dir)
        
        # Gestor de reportes
        self.report_manager = get_report_manager()
        
        # Estado actual
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        
        # Configuración de reportes
        self.report_config = {
            'send_on_phone_3s': True,
            'send_on_phone_7s': True,
            'send_on_smoking': True,
            'include_frame': True,
            'cooldown_seconds': 30
        }
        
        # Control de reportes
        self.last_report_time = {}
        
        # Historial para análisis
        self.detection_history = []
        self.max_history_size = 300
        
        # Estadísticas de sesión
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_phone_alerts': 0,
            'total_smoking_alerts': 0,
            'operators_monitored': set()
        }
        
        # Inicializar detector
        if not self.detector.initialize():
            self.logger.error("Error inicializando detector de comportamientos")
    
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
        self.detector.update_config(thresholds)
        
        # Resetear historial
        self.detection_history.clear()
        
        return True
    
    def analyze_frame(self, frame, face_locations=None):
        """
        Analiza un frame para detectar comportamientos.
        
        Args:
            frame: Frame de video
            face_locations: Ubicaciones de rostros detectados
            
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
        
        # Realizar detección
        detections, analyzed_frame, alerts = self.detector.detect_behaviors(frame, face_locations)
        
        # Procesar alertas y generar reportes
        for alert in alerts:
            self._handle_behavior_alert(alert, frame)
        
        # Crear resultado estructurado
        result = {
            'detections': detections,
            'alerts': alerts,
            'frame': analyzed_frame,
            'timestamp': time.time(),
            'operator_id': self.current_operator['id'],
            'operator_name': self.current_operator.get('name', 'Unknown'),
            'is_calibrated': self.is_calibrated,
            'is_night_mode': self.detector.is_night_mode,
            'light_level': self.detector.light_level,
            'optimization_status': self.detector.get_optimization_status()
        }
        
        # Agregar al historial
        self._update_history(result)
        
        return result
    
    def _handle_behavior_alert(self, alert, frame):
        """
        Maneja una alerta de comportamiento y genera reporte si necesario.
        
        Args:
            alert: Tupla (alert_type, behavior, value)
            frame: Frame actual
        """
        alert_type, behavior, value = alert
        current_time = time.time()
        
        # Verificar cooldown
        last_time = self.last_report_time.get(alert_type, 0)
        if current_time - last_time < self.report_config['cooldown_seconds']:
            return
        
        # Generar reporte según tipo de alerta
        should_report = False
        event_data = {}
        
        if alert_type == "phone_3s" and self.report_config['send_on_phone_3s']:
            should_report = True
            event_data = {
                'behavior': behavior,
                'duration': float(value),
                'threshold': 3,
                'severity': 'warning'
            }
            self.session_stats['total_phone_alerts'] += 1
            
        elif alert_type == "phone_7s" and self.report_config['send_on_phone_7s']:
            should_report = True
            event_data = {
                'behavior': behavior,
                'duration': float(value),
                'threshold': 7,
                'severity': 'critical'
            }
            
        elif alert_type in ["smoking_pattern", "smoking_7s"] and self.report_config['send_on_smoking']:
            should_report = True
            event_data = {
                'behavior': behavior,
                'detection_count': int(value) if alert_type == "smoking_pattern" else None,
                'duration': float(value) if alert_type == "smoking_7s" else None,
                'pattern_type': alert_type,
                'severity': 'warning'
            }
            self.session_stats['total_smoking_alerts'] += 1
        
        # Generar reporte
        if should_report:
            # Agregar información común
            event_data.update({
                'is_night_mode': bool(self.detector.is_night_mode),  # Convertir bool_ a bool
                'light_level': float(self.detector.light_level),     # Convertir a float nativo
                'detection_timestamp': float(current_time)            # Asegurar float nativo
            })
            
            report = self.report_manager.generate_report(
                module_name='behavior',
                event_type=alert_type,
                data=event_data,
                frame=frame if self.report_config['include_frame'] else None,
                operator_info=self.current_operator
            )
            
            if report:
                self.last_report_time[alert_type] = current_time
                self.logger.info(f"Reporte de comportamiento generado: {report['id']}")
    
    def _update_history(self, result):
        """Actualiza el historial de detecciones"""
        # Agregar solo datos relevantes
        history_entry = {
            'timestamp': result['timestamp'],
            'detections': len(result['detections']),
            'has_phone': any(d[0] == 'cell phone' for d in result['detections']),
            'has_cigarette': any(d[0] == 'cigarette' for d in result['detections']),
            'alerts_count': len(result['alerts'])
        }
        
        self.detection_history.append(history_entry)
        
        # Limitar tamaño del historial
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def get_current_status(self):
        """Obtiene el estado actual del sistema"""
        return {
            'operator': self.current_operator,
            'is_calibrated': self.is_calibrated,
            'thresholds': self.current_thresholds,
            'detector_config': self.detector.get_config(),
            'session_stats': self.session_stats,
            'history_size': len(self.detection_history),
            'report_stats': self.report_manager.get_statistics()
        }
    
    def generate_session_report(self):
        """Genera reporte de sesión completa"""
        try:
            session_duration = time.time() - self.session_stats['session_start']
            
            # Calcular estadísticas de comportamientos
            phone_detections = sum(1 for h in self.detection_history if h.get('has_phone', False))
            smoking_detections = sum(1 for h in self.detection_history if h.get('has_cigarette', False))
            
            session_data = {
                'session_start': self.session_stats['session_start'],
                'session_end': time.time(),
                'duration_seconds': session_duration,
                'operators_monitored': list(self.session_stats['operators_monitored']),
                'total_detections': self.session_stats['total_detections'],
                'total_phone_alerts': self.session_stats['total_phone_alerts'],
                'total_smoking_alerts': self.session_stats['total_smoking_alerts'],
                'behavior_summary': {
                    'phone_detection_rate': phone_detections / max(self.session_stats['total_detections'], 1),
                    'smoking_detection_rate': smoking_detections / max(self.session_stats['total_detections'], 1)
                },
                'calibration_used': self.is_calibrated
            }
            
            # Generar reporte
            report = self.report_manager.generate_report(
                module_name='behavior',
                event_type='session_summary',
                data=session_data,
                operator_info=self.current_operator
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de sesión: {e}")
            return None
    
    def _save_operator_session(self):
        """Guarda estadísticas de la sesión del operador"""
        if self.current_operator:
            self.logger.info(
                f"Sesión finalizada para {self.current_operator.get('name', 'Unknown')} - "
                f"Alertas: Teléfono={self.session_stats['total_phone_alerts']}, "
                f"Cigarrillo={self.session_stats['total_smoking_alerts']}"
            )
    
    def update_report_config(self, new_config):
        """Actualiza configuración de reportes"""
        self.report_config.update(new_config)
        self.logger.info("Configuración de reportes actualizada")
    
    def reset(self):
        """Reinicia el sistema completamente"""
        self.current_operator = None
        self.current_thresholds = None
        self.is_calibrated = False
        self.detection_history.clear()
        self.last_report_time.clear()
        
        # Reiniciar estadísticas
        self.session_stats = {
            'session_start': time.time(),
            'total_detections': 0,
            'total_phone_alerts': 0,
            'total_smoking_alerts': 0,
            'operators_monitored': set()
        }
        
        self.logger.info("Sistema de comportamientos reiniciado")