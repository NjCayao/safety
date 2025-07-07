import os
import time
import uuid
import shutil
import logging
from datetime import datetime
from configparser import ConfigParser

# Cargar configuración
config = ConfigParser()
config.read('config/config.ini')

logger = logging.getLogger('file_manager')

class FileManager:
    def __init__(self):
        self.base_path = config.get('STORAGE', 'image_storage_path')
        self.max_stored_images = config.getint('STORAGE', 'max_stored_images')
        
        # Crear directorio base si no existe
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
    
    def save_image(self, image_data, event_type):
        """Guardar imagen en disco"""
        try:
            # Crear directorio para tipo de evento
            event_dir = os.path.join(self.base_path, event_type)
            if not os.path.exists(event_dir):
                os.makedirs(event_dir, exist_ok=True)
            
            # Crear directorio para fecha actual
            date_dir = os.path.join(event_dir, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(date_dir):
                os.makedirs(date_dir, exist_ok=True)
            
            # Generar nombre único
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
            file_path = os.path.join(date_dir, filename)
            
            # Guardar archivo
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Imagen guardada: {file_path}")
            
            # Limpiar imágenes antiguas si es necesario
            self.cleanup_old_images()
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error al guardar imagen: {str(e)}")
            return None
    
    def cleanup_old_images(self):
        """Limpiar imágenes antiguas para ahorrar espacio"""
        try:
            # Obtener lista de todas las imágenes
            all_images = []
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        mtime = os.path.getmtime(full_path)
                        all_images.append((full_path, mtime))
            
            # Ordenar por fecha de modificación (más antiguo primero)
            all_images.sort(key=lambda x: x[1])
            
            # Eliminar imágenes antiguas si excedemos el límite
            if len(all_images) > self.max_stored_images:
                to_delete = all_images[:len(all_images) - self.max_stored_images]
                
                for path, _ in to_delete:
                    os.remove(path)
                    logger.info(f"Imagen antigua eliminada: {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error durante limpieza de imágenes: {str(e)}")
            return False
    
    def get_image_path(self, event_id, event_type):
        """Obtener ruta de imagen basada en ID de evento"""
        try:
            event_dir = os.path.join(self.base_path, event_type)
            
            # Buscar recursivamente
            for root, dirs, files in os.walk(event_dir):
                for file in files:
                    if file.startswith(event_id):
                        return os.path.join(root, file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error al buscar imagen: {str(e)}")
            return None