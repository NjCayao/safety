import time
import logging

# Importar el módulo original
from behavior_detection_module import BehaviorDetector
from client.utils.event_manager import EventManager

logger = logging.getLogger('behavior_adapter')

class BehaviorAdapter:
    """
    Adaptador para el detector de comportamientos que conecta con el sistema de sincronización.
    """
    def __init__(self):
        # Inicializar el detector original
        self.detector = BehaviorDetector()
        
        # Inicializar gestor de eventos
        self.event_manager = EventManager()
        
        # Control de eventos
        self.last_cellphone_event = 0
        self.last_smoking_event = 0
        self.min_time_between_events = 60  # segundos
    
    def detect_cellphone_and_sync(self, frame, operator_id=None):
        """
        Detecta uso de celular y registra el evento para sincronización.
        
        Args:
            frame: Imagen capturada de la cámara
            operator_id: ID del operador actual (opcional)
            
        Returns:
            dict: Resultado de la detección con información adicional
        """
        # Usar el detector original
        result = self.detector.detect_cellphone(frame)
        
        # Si se detecta uso de celular
        if result.get('cellphone_detected', False):
            # Evitar eventos duplicados
            if time.time() - self.last_cellphone_event > self.min_time_between_events:
                self.last_cellphone_event = time.time()
                
                # Preparar datos del evento
                event_data = {
                    'confidence': result.get('confidence', 0.0),
                    'bounding_box': result.get('bounding_box', [0, 0, 0, 0]),
                    'detection_time': time.time()
                }
                
                # Registrar evento para sincronización
                event_id = self.event_manager.register_event(
                    event_type='cellphone',
                    event_data=event_data,
                    frame=frame,
                    operator_id=operator_id
                )
                
                # Añadir ID del evento al resultado
                result['event_id'] = event_id
                logger.warning(f"Uso de celular detectado. Evento registrado con ID: {event_id}")
        
        return result
    
    def detect_smoking_and_sync(self, frame, operator_id=None):
        """
        Detecta si está fumando y registra el evento para sincronización.
        
        Args:
            frame: Imagen capturada de la cámara
            operator_id: ID del operador actual (opcional)
            
        Returns:
            dict: Resultado de la detección con información adicional
        """
        # Usar el detector original
        result = self.detector.detect_smoking(frame)
        
        # Si se detecta que está fumando
        if result.get('smoking_detected', False):
            # Evitar eventos duplicados
            if time.time() - self.last_smoking_event > self.min_time_between_events:
                self.last_smoking_event = time.time()
                
                # Preparar datos del evento
                event_data = {
                    'confidence': result.get('confidence', 0.0),
                    'bounding_box': result.get('bounding_box', [0, 0, 0, 0]),
                    'detection_time': time.time()
                }
                
                # Registrar evento para sincronización
                event_id = self.event_manager.register_event(
                    event_type='smoking',
                    event_data=event_data,
                    frame=frame,
                    operator_id=operator_id
                )
                
                # Añadir ID del evento al resultado
                result['event_id'] = event_id
                logger.warning(f"Fumando detectado. Evento registrado con ID: {event_id}")
        
        return result
    
    # Método para acceder directamente al detector original
    def get_detector(self):
        return self.detector