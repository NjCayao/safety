import time
import logging

# Importar el módulo original
from face_recognition import FaceRecognizer
from client.utils.event_manager import EventManager

logger = logging.getLogger('face_recognition_adapter')

class FaceRecognitionAdapter:
    """
    Adaptador para el reconocedor facial que conecta con el sistema de sincronización.
    """
    def __init__(self):
        # Inicializar el reconocedor original
        self.recognizer = FaceRecognizer()
        
        # Inicializar gestor de eventos
        self.event_manager = EventManager()
        
        # Control de eventos
        self.last_unrecognized_event = 0
        self.min_time_between_events = 60  # segundos
    
    def recognize_and_sync(self, frame):
        """
        Reconoce la cara del operador y registra eventos de operador no reconocido.
        
        Args:
            frame: Imagen capturada de la cámara
            
        Returns:
            dict: Resultado del reconocimiento con información adicional
        """
        # Usar el reconocedor original
        result = self.recognizer.recognize_face(frame)
        
        # Si no se reconoce al operador
        if not result.get('recognized', False):
            # Evitar eventos duplicados
            if time.time() - self.last_unrecognized_event > self.min_time_between_events:
                self.last_unrecognized_event = time.time()
                
                # Preparar datos del evento
                event_data = {
                    'face_detected': result.get('face_detected', False),
                    'confidence': result.get('confidence', 0.0),
                    'detection_time': time.time()
                }
                
                # Registrar evento para sincronización
                event_id = self.event_manager.register_event(
                    event_type='unrecognized_operator',
                    event_data=event_data,
                    frame=frame
                )
                
                # Añadir ID del evento al resultado
                result['event_id'] = event_id
                logger.warning(f"Operador no reconocido. Evento registrado con ID: {event_id}")
        
        return result
    
    # Método para acceder directamente al reconocedor original
    def get_recognizer(self):
        return self.recognizer