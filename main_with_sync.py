#!/usr/bin/env python3
import time
import signal
import sys
import logging

# Importar componentes
from main_system import SafetySystem
from sync_integrator import SyncIntegrator

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('safety_system_with_sync.log')
    ]
)

logger = logging.getLogger('main_with_sync')

class SafetySystemWithSync:
    """
    Sistema de prevención de accidentes con sincronización online/offline.
    Este sistema es un ejemplo de cómo integrar el sistema principal con
    la funcionalidad de sincronización.
    """
    def __init__(self):
        logger.info("Iniciando Sistema de Prevención de Accidentes con Sincronización")
        
        # Inicializar integrador de sincronización
        self.sync_integrator = SyncIntegrator()
        
        # Inicializar sistema principal
        self.safety_system = SafetySystem()
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        
        self.running = False
    
    def start(self):
        """Iniciar el sistema completo"""
        logger.info("Iniciando sistema completo...")
        
        # Iniciar integrador de sincronización
        self.sync_integrator.start()
        
        # Aquí puedes modificar cómo el sistema principal usa los adaptadores
        # Por ejemplo, podrías modificar el bucle principal para usar los adaptadores
        # en lugar de los detectores originales
        
        # Iniciar sistema principal
        self.safety_system.start()
        
        self.running = True
        logger.info("Sistema completo iniciado")
        
        # Imprimir instrucciones de uso
        print("\nSistema de Prevención de Accidentes con Sincronización Online/Offline")
        print("-------------------------------------------------------------------")
        print("El sistema está ejecutándose con la funcionalidad de sincronización.")
        print("Los eventos detectados se almacenarán localmente cuando no haya conexión")
        print("y se sincronizarán automáticamente cuando la conexión se restablezca.")
        print("\nPresione Ctrl+C para detener el sistema.\n")
        
        return True
    
    def handle_exit(self, signum, frame):
        """Manejar señales de salida"""
        print("\nCerrando el sistema...")
        self.stop()
    
    def stop(self):
        """Detener el sistema completo"""
        if not self.running:
            return
            
        self.running = False
        
        # Detener sistema principal
        logger.info("Deteniendo sistema principal...")
        self.safety_system.stop()
        
        # Detener integrador de sincronización
        logger.info("Deteniendo integrador de sincronización...")
        self.sync_integrator.stop()
        
        logger.info("Sistema completo detenido")
        
        # Salir del programa
        sys.exit(0)

# Punto de entrada
if __name__ == "__main__":
    system = SafetySystemWithSync()
    try:
        if system.start():
            # Mantener el programa principal ejecutándose
            while True:
                # Mostrar estado de sincronización cada 5 minutos
                if int(time.time()) % 300 == 0:  # cada 5 minutos
                    status = system.sync_integrator.get_sync_status()
                    logger.info(f"Estado de sincronización: "
                              f"{'ONLINE' if status['is_online'] else 'OFFLINE'}, "
                              f"{status['pending_events']} eventos pendientes, "
                              f"{'Sincronizando' if status['is_syncing'] else 'No sincronizando'}")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()