import time
import threading
import logging
import sys
import signal

# Importar el sistema original
from main_system import SafetySystem as OriginalSafetySystem

# Importar componentes de sincronización
from client.utils.logger import setup_logging
from client.db.local_storage import LocalStorage
from client.api.api_client import ApiClient
from client.utils.connection import ConnectionManager
from client.api.sync import SyncManager
from client.utils.file_manager import FileManager

# Importar wrappers
from sync.wrappers.behavior_detection_wrapper import BehaviorDetectorWrapper
from sync.wrappers.fatigue_detection_wrapper import FatigueDetectorWrapper
from sync.wrappers.face_recognition_wrapper import FaceRecognizerWrapper

logger = logging.getLogger('main_system_wrapper')

class SafetySystemWrapper:
    """
    Wrapper para el sistema principal que añade funcionalidad
    de sincronización online/offline.
    """
    def __init__(self):
        # Inicializar logger
        self.logger = setup_logging()
        self.logger.info("Iniciando Sistema de Seguridad con Sincronización")
        
        # Inicializar componentes de sincronización
        self.db = LocalStorage()
        self.api_client = ApiClient()
        self.connection_manager = ConnectionManager(self.db, self.api_client)
        self.sync_manager = SyncManager(self.db, self.api_client, self.connection_manager)
        self.file_manager = FileManager()
        
        # Inicializar sistema original
        self.original_system = OriginalSafetySystem()
        
        # Reemplazar detectores con wrappers
        self.original_system.behavior_detector = BehaviorDetectorWrapper()
        self.original_system.fatigue_detector = FatigueDetectorWrapper()
        self.original_system.face_recognizer = FaceRecognizerWrapper()
        
        # Estado del sistema
        self.running = False
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
    def start(self):
        """Iniciar todos los servicios"""
        self.running = True
        
        # Iniciar monitoreo de conexión
        self.connection_manager.start_monitoring()
        self.logger.info("Monitoreo de conexión iniciado")
        
        # Iniciar sincronización automática
        self.sync_manager.start_auto_sync()
        self.logger.info("Sincronización automática iniciada")
        
        # Iniciar sistema original
        self.original_system.start()
        self.logger.info("Sistema principal iniciado")
        
        return True
    
    def handle_exit(self, signum, frame):
        """Manejar señales de salida"""
        self.logger.info("Señal de salida recibida. Cerrando sistema...")
        self.stop()
    
    def stop(self):
        """Detener todos los servicios"""
        self.running = False
        
        # Detener sistema original
        self.original_system.stop()
        self.logger.info("Sistema principal detenido")
        
        # Detener servicios de sincronización
        self.sync_manager.stop_auto_sync()
        self.logger.info("Sincronización automática detenida")
        
        self.connection_manager.stop_monitoring()
        self.logger.info("Monitoreo de conexión detenido")
        
        # Cerrar conexiones
        self.db.close()
        self.logger.info("Conexiones cerradas")
        
        self.logger.info("Sistema detenido correctamente")
        
        # Salir del programa
        sys.exit(0)

# Punto de entrada
if __name__ == "__main__":
    safety_system = SafetySystemWrapper()
    try:
        if safety_system.start():
            # Mantener el programa principal ejecutándose
            while safety_system.running:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nCerrando el sistema...")
    finally:
        safety_system.stop()