import os
import cv2
import time
import logging
import uuid
from client.db.local_storage import LocalStorage
from client.utils.file_manager import FileManager

logger = logging.getLogger('event_manager')

class EventManager:
    """
    Gestor centralizado de eventos detectados por los diferentes módulos.
    Maneja el almacenamiento local y prepara los eventos para sincronización.
    """
    def __init__(self):
        self.db = LocalStorage()
        self.file_manager = FileManager()
    
    def register_event(self, event_type, event_data, frame=None, operator_id=None):
        """
        Registra un nuevo evento en el sistema.
        
        Args:
            event_type (str): Tipo de evento ('fatigue', 'cellphone', 'smoking', 'unrecognized_operator')
            event_data (dict): Datos específicos del evento
            frame (numpy.ndarray, optional): Imagen capturada del evento
            operator_id (int, optional): ID del operador implicado
            
        Returns:
            str: ID del evento registrado o None si falló
        """
        try:
            # Guardar imagen si está disponible
            image_path = None
            if frame is not None:
                # Convertir frame a formato de imagen
                _, img_encoded = cv2.imencode('.jpg', frame)
                # Guardar imagen localmente
                image_path = self.file_manager.save_image(img_encoded.tobytes(), event_type)
                
                if image_path:
                    logger.info(f"Imagen guardada para evento {event_type}: {image_path}")
                else:
                    logger.warning(f"No se pudo guardar la imagen para evento {event_type}")
            
            # Registrar evento en la base de datos local
            event_id = self.db.store_event(
                event_type=event_type,
                event_data=event_data,
                image_path=image_path,
                operator_id=operator_id
            )
            
            if event_id:
                logger.info(f"Evento {event_type} registrado con ID: {event_id}")
                return event_id
            else:
                logger.error(f"Error al registrar evento {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Excepción al registrar evento {event_type}: {str(e)}")
            return None