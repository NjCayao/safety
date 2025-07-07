import time
import logging
from core.fatigue.fatigue_detection import FatigueDetector as OriginalFatigueDetector
from client.utils.event_manager import EventManager

logger = logging.getLogger('fatigue_detector_wrapper')

class FatigueDetectorWrapper:
    """
    Wrapper para el detector de fatiga que añade funcionalidad
    de almacenamiento local y sincronización.
    """
    def __init__(self):
        # Inicializar el detector original
        self.original_detector = OriginalFatigueDetector()
        
        # Inicializar gestor de eventos
        self.event_manager = EventManager()
        
        # Control de eventos
        self.last_fatigue_event = 0
        self.min_time_between_events = 30  # segundos
    
    def detect_fatigue(self, frame, operator_id=None):
        """
        Detecta fatiga en el operador.
        
        Args:
            frame: Imagen capturada de la cámara
            operator_id: ID del operador actual (opcional)
            
        Returns:
            dict: Resultado de la detección con información adicional
        """
        # Usar el detector original
        result = self.original_detector.detect_fatigue(frame)
        
        # Si se detecta fatiga
        if result.get('fatigue_detected', False):
            # Evitar eventos duplicados
            if time.time() - self.last_fatigue_event > self.min_time_between_events:
                self.last_fatigue_event = time.time()
                
                # Preparar datos del evento
                event_data = {
                    'fatigue_level': result.get('fatigue_level', 'medium'),
                    'eyes_closed_duration': result.get('eyes_closed_duration', 0),
                    'ear_value': result.get('ear_value', 0),
                    'detection_time': time.time()
                }
                
                # Registrar evento para sincronización
                event_id = self.event_manager.register_event(
                    event_type='fatigue',
                    event_data=event_data,
                    frame=frame,
                    operator_id=operator_id
                )
                
                # Añadir ID del evento al resultado
                result['event_id'] = event_id
                logger.warning(f"Fatiga detectada. Evento registrado con ID: {event_id}")
        
        return result
    
    # Añadir otros métodos del detector original según sea necesario,
    # manteniendo la misma interfaz para compatibilidad