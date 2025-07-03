import time
import threading
import logging
import signal
import sys

# Importar el sistema principal
from main_system import SafetySystem

# Importar componentes de sincronización
from client.utils.logger import setup_logging
from client.db.local_storage import LocalStorage
from client.api.api_client import ApiClient
from client.utils.connection import ConnectionManager
from client.api.sync import SyncManager

# Importar adaptadores
from fatigue_adapter import FatigueAdapter
from behavior_adapter import BehaviorAdapter
from face_recognition_adapter import FaceRecognitionAdapter

logger = logging.getLogger('sync_integrator')

class SyncIntegrator:
    """
    Integrador que añade sincronización online/offline al sistema principal.
    """
    def __init__(self):
        # Inicializar logger
        setup_logging()
        logger.info("Iniciando Integrador de Sincronización")
        
        # Inicializar componentes de sincronización
        self.db = LocalStorage()
        self.api_client = ApiClient()
        self.connection_manager = ConnectionManager(self.db, self.api_client)
        self.sync_manager = SyncManager(self.db, self.api_client, self.connection_manager)
        
        # Inicializar adaptadores
        self.fatigue_adapter = FatigueAdapter()
        self.behavior_adapter = BehaviorAdapter()
        self.face_recognition_adapter = FaceRecognitionAdapter()
        
        # Sistema principal (mantener referencia si es necesario)
        self.safety_system = None
        
        # Estado del integrador
        self.running = False
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
    def start(self):
        """
        Inicia los servicios de sincronización.
        Este método debe llamarse antes de iniciar el sistema principal.
        """
        logger.info("Iniciando servicios de sincronización")
        
        # Iniciar monitoreo de conexión
        self.connection_manager.start_monitoring()
        logger.info("Monitoreo de conexión iniciado")
        
        # Iniciar sincronización automática
        self.sync_manager.start_auto_sync()
        logger.info("Sincronización automática iniciada")
        
        self.running = True
        logger.info("Integrador de sincronización iniciado correctamente")
        
        return True
    
    def handle_exit(self, signum, frame):
        """Manejar señales de salida"""
        logger.info("Señal de salida recibida. Cerrando sistema de sincronización...")
        self.stop()
    
    def stop(self):
        """Detener los servicios de sincronización"""
        if not self.running:
            return
            
        self.running = False
        
        # Detener servicios de sincronización
        logger.info("Deteniendo servicios de sincronización...")
        
        self.sync_manager.stop_auto_sync()
        logger.info("Sincronización automática detenida")
        
        self.connection_manager.stop_monitoring()
        logger.info("Monitoreo de conexión detenido")
        
        # Cerrar conexiones
        self.db.close()
        logger.info("Conexiones cerradas")
        
        logger.info("Integrador de sincronización detenido correctamente")
    
    def get_fatigue_adapter(self):
        """Obtener el adaptador de fatiga"""
        return self.fatigue_adapter
    
    def get_behavior_adapter(self):
        """Obtener el adaptador de comportamientos"""
        return self.behavior_adapter
    
    def get_face_recognition_adapter(self):
        """Obtener el adaptador de reconocimiento facial"""
        return self.face_recognition_adapter
    
    # Método para verificar estado de sincronización
    def get_sync_status(self):
        """
        Obtener estado actual de la sincronización.
        
        Returns:
            dict: Estado de la sincronización
        """
        return {
            'is_online': self.connection_manager.is_online(),
            'pending_events': len(self.db.get_pending_events(1000000)),
            'is_syncing': self.sync_manager.is_syncing,
            'last_sync': self.db.get_connection_status().get('last_sync')
        }