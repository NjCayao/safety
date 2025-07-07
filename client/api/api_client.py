import requests
import json
import os
import time
import logging
from configparser import ConfigParser

# Cargar configuración
config = ConfigParser()
config.read('config/config.ini')

logger = logging.getLogger('api_client')

class ApiClient:
    def __init__(self):
        self.base_url = config.get('SERVER', 'api_url')
        self.device_id = config.get('DEVICE', 'device_id')
        self.api_key = config.get('DEVICE', 'api_key')
        self.token = None
        self.token_expiration = 0
        self.retry_attempts = config.getint('CONNECTION', 'retry_attempts')
        self.retry_delay = config.getint('CONNECTION', 'retry_delay')
    
    def authenticate(self):
        """Autenticar con el servidor y obtener token JWT"""
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'auth_endpoint')}"
            payload = {
                "device_id": self.device_id,
                "api_key": self.api_key
            }
            
            response = self._make_request('POST', endpoint, json=payload)
            
            if response and response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.token = data['data']['token']
                    # Calcular tiempo de expiración (10 minutos antes para margen de seguridad)
                    self.token_expiration = time.time() + data['data']['expires_in'] - 600
                    logger.info(f"Autenticación exitosa. Token válido hasta: {time.ctime(self.token_expiration)}")
                    return True
            
            logger.error(f"Error de autenticación: {response.text if response else 'Sin respuesta'}")
            return False
        
        except Exception as e:
            logger.error(f"Excepción durante autenticación: {str(e)}")
            return False
    
    def is_token_valid(self):
        """Verificar si el token actual es válido"""
        return self.token is not None and time.time() < self.token_expiration
    
    def ensure_authenticated(self):
        """Asegurar que el cliente está autenticado antes de hacer solicitudes"""
        if not self.is_token_valid():
            return self.authenticate()
        return True
    
    def create_event(self, event_data, image_path=None):
        """Crear un nuevo evento en el servidor"""
        if not self.ensure_authenticated():
            return False, "Error de autenticación"
        
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'events_endpoint')}"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            # Determinar si hay imagen
            has_image = image_path is not None and os.path.exists(image_path)
            event_data['has_image'] = has_image
            
            # Enviar datos del evento
            response = self._make_request('POST', endpoint, headers=headers, json=event_data)
            
            if not response or response.status_code != 201:
                return False, f"Error al crear evento: {response.text if response else 'Sin respuesta'}"
            
            event_response = response.json()
            
            # Si hay imagen, subirla
            if has_image and event_response.get('status') == 'success':
                event_id = event_response['data']['event_id']
                image_uploaded = self.upload_image(event_id, image_path)
                if not image_uploaded:
                    logger.warning(f"Evento creado pero imagen no subida para evento {event_id}")
            
            return True, event_response.get('data', {})
            
        except Exception as e:
            logger.error(f"Excepción al crear evento: {str(e)}")
            return False, str(e)
    
    def upload_image(self, event_id, image_path):
        """Subir imagen para un evento específico"""
        if not self.ensure_authenticated():
            return False
        
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'upload_image_endpoint')}"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'event_id': event_id}
                
                response = self._make_request('POST', endpoint, headers=headers, data=data, files=files)
                
                if response and response.status_code == 200:
                    logger.info(f"Imagen subida exitosamente para evento {event_id}")
                    return True
                
                logger.error(f"Error al subir imagen: {response.text if response else 'Sin respuesta'}")
                return False
                
        except Exception as e:
            logger.error(f"Excepción al subir imagen: {str(e)}")
            return False
    
    def sync_batch(self, batch_id, events):
        """Sincronizar un lote de eventos"""
        if not self.ensure_authenticated():
            return False, "Error de autenticación"
        
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'sync_batch_endpoint')}"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            payload = {
                "batch_id": batch_id,
                "events": events
            }
            
            response = self._make_request('POST', endpoint, headers=headers, json=payload)
            
            if not response:
                return False, "Sin respuesta del servidor"
                
            if response.status_code in [200, 206]:
                return True, response.json()
            
            return False, f"Error al sincronizar lote: {response.text}"
            
        except Exception as e:
            logger.error(f"Excepción al sincronizar lote: {str(e)}")
            return False, str(e)
    
    def get_sync_status(self):
        """Obtener estado de sincronización"""
        if not self.ensure_authenticated():
            return None
        
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'sync_status_endpoint')}"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = self._make_request('GET', endpoint, headers=headers)
            
            if response and response.status_code == 200:
                return response.json().get('data')
            
            return None
            
        except Exception as e:
            logger.error(f"Excepción al obtener estado de sincronización: {str(e)}")
            return None
    
    def confirm_sync(self, batch_id):
        """Confirmar sincronización exitosa de un lote"""
        if not self.ensure_authenticated():
            return False
        
        try:
            endpoint = f"{self.base_url}{config.get('SERVER', 'sync_confirm_endpoint')}"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            payload = {
                "batch_id": batch_id
            }
            
            response = self._make_request('POST', endpoint, headers=headers, json=payload)
            
            return response and response.status_code == 200
            
        except Exception as e:
            logger.error(f"Excepción al confirmar sincronización: {str(e)}")
            return False
    
    def _make_request(self, method, url, **kwargs):
        """Realizar solicitud HTTP con reintentos"""
        for attempt in range(self.retry_attempts + 1):
            try:
                response = requests.request(method, url, **kwargs)
                return response
            except requests.RequestException as e:
                if attempt < self.retry_attempts:
                    logger.warning(f"Intento {attempt+1} fallido: {str(e)}. Reintentando en {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Todos los intentos fallidos para {url}: {str(e)}")
                    return None