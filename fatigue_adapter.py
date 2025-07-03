import cv2
import time
import logging

# Importar el módulo original
from fatigue_detection import FatigueDetector
from client.utils.event_manager import EventManager

logger = logging.getLogger('fatigue_adapter')

class FatigueAdapter:
    """
    Adaptador para el detector de fatiga que conecta con el sistema de sincronización.
    """
    def __init__(self):
        # Inicializar el detector original
        self.detector = FatigueDetector()
        
        # Inicializar gestor de eventos
        self.event_manager = EventManager()
        
        # Control de eventos
        self.last_fatigue_event = 0
        self.min_time_between_events = 30  # segundos
    
    def detect_and_sync(self, frame, operator_id=None):
        """
        Detecta fatiga y registra el evento para sincronización.
        
        Args:
            frame: Imagen capturada de la cámara
            operator_id: ID del operador actual (opcional)
            
        Returns:
            dict: Resultado de la detección con información adicional
        """
        # Usar el detector original
        result = self.detector.detect_fatigue(frame)
        
        # Si se detecta fatiga
        if result.get('fatigue_detected', False):
            # Evitar eventos duplicados
            if time.time() - self.last_fatigue_event > self.min_time_between_events:
                self.last_fatigue_event = time.time()
                
                # Preparar datos del evento
                event_data = {
                    'fatigue_level': result.get('fatigue_level', 'medium'),
                    'eyes_closed_duration': result.get('eyes_closed_duration', 0),
                    'confidence': result.get('confidence', 0.8),
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
    
    # Método para acceder directamente al detector original
    def get_detector(self):
        return self.detector