import requests
import time
import logging
import threading
from configparser import ConfigParser

# Cargar configuración
config = ConfigParser()
config.read('config/config.ini')

logger = logging.getLogger('connection')

class ConnectionManager:
    def __init__(self, db, api_client):
        self.db = db
        self.api_client = api_client
        self.check_interval = config.getint('CONNECTION', 'check_interval')
        self.is_running = False
        self.thread = None
    
    def start_monitoring(self):
        """Iniciar monitoreo de conexión en un hilo separado"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Monitoreo de conexión iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de conexión"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
            logger.info("Monitoreo de conexión detenido")
    
    def _monitor_loop(self):
        """Bucle principal de monitoreo"""
        while self.is_running:
            is_online = self.check_connection()
            self.db.update_connection_status(is_online)
            
            # Si estamos en línea, intentar autenticar
            if is_online and not self.api_client.is_token_valid():
                self.api_client.authenticate()
            
            time.sleep(self.check_interval)
    
    def check_connection(self):
        """Verificar si hay conexión a internet"""
        try:
            # Intentar conectar a un servicio confiable
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_status(self):
        """Obtener estado actual de conexión"""
        return self.db.get_connection_status()
    
    def is_online(self):
        """Verificar si actualmente estamos en línea"""
        return self.check_connection()
    
    def wait_for_connection(self, timeout=None):
        """Esperar hasta que haya conexión o se alcance el tiempo de espera"""
        start_time = time.time()
        while not self.check_connection():
            if timeout and (time.time() - start_time > timeout):
                return False
            logger.info("Esperando conexión...")
            time.sleep(5)
        return True