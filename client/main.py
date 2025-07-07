import os
import time
import signal
import sys
from configparser import ConfigParser

# Importar componentes
from utils.logger import setup_logging
from db.local_storage import LocalStorage
from api.api_client import ApiClient
from utils.connection import ConnectionManager
from api.sync import SyncManager
from utils.file_manager import FileManager

# Configuración
config = ConfigParser()
config.read('config/config.ini')

# Inicializar logger
logger = setup_logging()

class SafetySystemClient:
    def __init__(self):
        self.running = True
        
        # Inicializar componentes
        logger.info("Iniciando sistema de seguridad...")
        
        self.db = LocalStorage()
        logger.info("Almacenamiento local inicializado")
        
        self.api_client = ApiClient()
        logger.info("Cliente API inicializado")
        
        self.connection_manager = ConnectionManager(self.db, self.api_client)
        logger.info("Gestor de conexión inicializado")
        
        self.sync_manager = SyncManager(self.db, self.api_client, self.connection_manager)
        logger.info("Gestor de sincronización inicializado")
        
        self.file_manager = FileManager()
        logger.info("Gestor de archivos inicializado")
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
    def start(self):
        """Iniciar todos los servicios"""
        # Iniciar monitoreo de conexión
        self.connection_manager.start_monitoring()
        
        # Esperar conexión inicial para intentar autenticación
        logger.info("Esperando conexión inicial...")
        if self.connection_manager.wait_for_connection(timeout=30):
            logger.info("Conexión establecida, autenticando...")
            self.api_client.authenticate()
        else:
            logger.warning("No se pudo establecer conexión inicial. Continuando en modo offline.")
        
        # Iniciar sincronización automática
        self.sync_manager.start_auto_sync()
        
        logger.info("Sistema iniciado completamente")
        
        # Bucle principal
        while self.running:
            try:
                # Aquí se integrarían los módulos de detección
                # Este es solo un ejemplo del bucle principal
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error en bucle principal: {str(e)}")
    
    def handle_exit(self, signum, frame):
        """Manejar señales de salida"""
        logger.info("Señal de salida recibida. Cerrando sistema...")
        self.stop()
    
    def stop(self):
        """Detener todos los servicios"""
        self.running = False
        
        # Detener servicios
        logger.info("Deteniendo servicios...")
        
        self.sync_manager.stop_auto_sync()
        logger.info("Sincronización automática detenida")
        
        self.connection_manager.stop_monitoring()
        logger.info("Monitoreo de conexión detenido")
        
        # Cerrar conexiones
        self.db.close()
        logger.info("Conexiones cerradas")
        
        logger.info("Sistema detenido correctamente")

# Punto de entrada
if __name__ == "__main__":
    system = SafetySystemClient()
    try:
        system.start()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()