# sync/heartbeat_sender.py
"""
Enviador de heartbeats al servidor
Mantiene conexión activa y reporta estado del sistema
"""

import time
import json
import logging
import requests
import threading
import platform
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from config.config_manager import get_config
from sync.device_auth import get_device_authenticator

class HeartbeatSender:
    """
    Envía heartbeats periódicos al servidor para mantener el estado de conexión
    y reportar información del sistema.
    """
    
    def __init__(self):
        self.authenticator = get_device_authenticator()
        self.logger = logging.getLogger('HeartbeatSender')
        
        # Configuración
        self.server_url = get_config('sync.server_url', 'http://localhost/safety_system')
        self.heartbeat_interval = get_config('sync.heartbeat_interval', 120)  # 2 minutos
        self.heartbeat_timeout = get_config('sync.heartbeat_timeout', 15)
        
        # Estado interno
        self.is_running = False
        self._heartbeat_thread = None
        self._stop_event = threading.Event()
        self._last_heartbeat = None
        self._consecutive_failures = 0
        
        # Endpoints
        self.heartbeat_endpoint = f"{self.server_url}/api/v1/devices/heartbeat"
        
        self.logger.info("Enviador de heartbeats inicializado")
    
    def start(self):
        """Inicia el envío de heartbeats en un hilo separado"""
        if self.is_running:
            self.logger.warning("Heartbeat sender ya está ejecutándose")
            return
        
        self.logger.info("Iniciando envío de heartbeats")
        self.is_running = True
        self._stop_event.clear()
        
        # Crear y iniciar hilo
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="HeartbeatSender",
            daemon=True
        )
        self._heartbeat_thread.start()
    
    def stop(self):
        """Detiene el envío de heartbeats"""
        if not self.is_running:
            return
        
        self.logger.info("Deteniendo envío de heartbeats")
        self.is_running = False
        self._stop_event.set()
        
        # Esperar a que termine el hilo
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
    
    def send_immediate_heartbeat(self) -> bool:
        """
        Envía un heartbeat inmediato.
        
        Returns:
            bool: True si se envió exitosamente
        """
        return self._send_heartbeat()
    
    def get_last_heartbeat_time(self) -> Optional[datetime]:
        """Retorna el timestamp del último heartbeat exitoso"""
        return self._last_heartbeat
    
    def get_consecutive_failures(self) -> int:
        """Retorna el número de fallos consecutivos"""
        return self._consecutive_failures
    
    def _heartbeat_loop(self):
        """Bucle principal de envío de heartbeats"""
        self.logger.info("Bucle de heartbeats iniciado")
        
        # Enviar heartbeat inicial inmediatamente
        self._send_heartbeat()
        
        while self.is_running and not self._stop_event.is_set():
            try:
                # Esperar hasta el próximo heartbeat
                if self._stop_event.wait(self.heartbeat_interval):
                    break  # Se solicitó parar
                
                # Enviar heartbeat
                self._send_heartbeat()
                
            except Exception as e:
                self.logger.error(f"Error en bucle de heartbeats: {e}")
                # Esperar un poco antes de continuar
                if self._stop_event.wait(30):
                    break
        
        self.logger.info("Bucle de heartbeats terminado")
    
    def _send_heartbeat(self) -> bool:
        """
        Envía un heartbeat al servidor.
        
        Returns:
            bool: True si se envió exitosamente
        """
        try:
            # Obtener headers de autenticación
            headers = self.authenticator.get_auth_headers()
            
            if 'Authorization' not in headers:
                self.logger.debug("No hay token de autenticación, saltando heartbeat")
                return False
            
            # Preparar datos del heartbeat
            heartbeat_data = {
                'device_id': self.authenticator.get_device_id(),
                'timestamp': datetime.now().isoformat(),
                'status': self._get_device_status(),
                'system_info': self._get_system_info(),
                'performance': self._get_performance_info()
            }
            
            # Enviar heartbeat
            response = requests.post(
                self.heartbeat_endpoint,
                headers=headers,
                json=heartbeat_data,
                timeout=self.heartbeat_timeout
            )
            
            if response.status_code == 200:
                # Heartbeat exitoso
                self._last_heartbeat = datetime.now()
                self._consecutive_failures = 0
                
                # Procesar respuesta del servidor
                self._process_heartbeat_response(response.json())
                
                self.logger.debug("Heartbeat enviado exitosamente")
                return True
                
            elif response.status_code == 401:
                # Token expirado
                self.logger.warning("Token expirado durante heartbeat")
                self.authenticator.refresh_token()
                self._consecutive_failures += 1
                return False
                
            else:
                # Error del servidor
                self.logger.warning(f"Error en heartbeat: {response.status_code}")
                self._consecutive_failures += 1
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Error de conexión en heartbeat: {e}")
            self._consecutive_failures += 1
            return False
        except Exception as e:
            self.logger.error(f"Error inesperado en heartbeat: {e}")
            self._consecutive_failures += 1
            return False
    
    def _get_device_status(self) -> str:
        """
        Determina el estado actual del dispositivo.
        
        Returns:
            str: Estado del dispositivo ('online', 'warning', 'error')
        """
        try:
            # Verificar si hay errores críticos
            if self._consecutive_failures > 5:
                return 'error'
            
            # Verificar recursos del sistema
            system_info = self._get_system_info()
            
            # Verificar uso de CPU y memoria
            cpu_usage = system_info.get('cpu_percent', 0)
            memory_usage = system_info.get('memory_percent', 0)
            disk_usage = system_info.get('disk_percent', 0)
            temperature = system_info.get('temperature')
            
            # Determinar estado basado en recursos
            if (cpu_usage > 90 or memory_usage > 90 or disk_usage > 95 or 
                (temperature and temperature > 75)):
                return 'warning'
            
            return 'online'
            
        except Exception as e:
            self.logger.warning(f"Error determinando estado del dispositivo: {e}")
            return 'error'
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema.
        
        Returns:
            Dict con información del sistema
        """
        try:
            import psutil
            
            # Información básica del sistema
            info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'architecture': platform.architecture()[0],
                'hostname': platform.node(),
                'uptime_seconds': time.time() - psutil.boot_time()
            }
            
            # Información de recursos
            info.update({
                'cpu_percent': round(psutil.cpu_percent(interval=1), 1),
                'cpu_count': psutil.cpu_count(),
                'memory_percent': round(psutil.virtual_memory().percent, 1),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_percent': round(psutil.disk_usage('/').percent, 1),
                'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2)
            })
            
            # Temperatura (solo en Raspberry Pi)
            temperature = self._get_cpu_temperature()
            if temperature:
                info['temperature'] = temperature
            
            # Información de red
            net_info = self._get_network_info()
            if net_info:
                info['network'] = net_info
            
            return info
            
        except ImportError:
            # psutil no disponible
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'error': 'psutil no disponible'
            }
        except Exception as e:
            self.logger.warning(f"Error obteniendo información del sistema: {e}")
            return {'error': str(e)}
    
    def _get_performance_info(self) -> Dict[str, Any]:
        """
        Obtiene información de rendimiento del sistema de detección.
        
        Returns:
            Dict con métricas de rendimiento
        """
        try:
            # Aquí podrías agregar métricas específicas de tu sistema
            # Por ejemplo: FPS de procesamiento, latencia de detección, etc.
            
            performance = {
                'config_version': get_config('system.config_version', 1),
                'detection_active': True,  # Esto lo determinarías desde tu sistema principal
                'last_config_check': self._get_last_config_check_time()
            }
            
            # Agregar métricas de archivos de log si existen
            log_info = self._get_log_info()
            if log_info:
                performance['logs'] = log_info
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Error obteniendo información de rendimiento: {e}")
            return {}
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """
        Obtiene la temperatura de la CPU (Raspberry Pi).
        
        Returns:
            float: Temperatura en Celsius o None si no está disponible
        """
        try:
            # Método para Raspberry Pi
            if Path('/sys/class/thermal/thermal_zone0/temp').exists():
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millis = int(f.read().strip())
                    return round(temp_millis / 1000.0, 1)
            
            # Método alternativo usando vcgencmd (Raspberry Pi)
            import subprocess
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                # Formato: temp=47.7'C
                if 'temp=' in temp_str:
                    temp = float(temp_str.split('=')[1].replace("'C", ""))
                    return round(temp, 1)
            
        except Exception as e:
            self.logger.debug(f"No se pudo obtener temperatura de CPU: {e}")
        
        return None
    
    def _get_network_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de red.
        
        Returns:
            Dict con información de red
        """
        try:
            import psutil
            
            # Obtener interfaces de red
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            network_info = {
                'interfaces': {},
                'connections': len(psutil.net_connections())
            }
            
            for interface, addresses in interfaces.items():
                if interface != 'lo':  # Ignorar loopback
                    interface_info = {
                        'is_up': stats.get(interface, {}).isup if interface in stats else False,
                        'addresses': []
                    }
                    
                    for addr in addresses:
                        if addr.family.name in ['AF_INET', 'AF_INET6']:
                            interface_info['addresses'].append({
                                'family': addr.family.name,
                                'address': addr.address
                            })
                    
                    if interface_info['addresses']:
                        network_info['interfaces'][interface] = interface_info
            
            return network_info
            
        except Exception as e:
            self.logger.debug(f"Error obteniendo información de red: {e}")
            return None
    
    def _get_last_config_check_time(self) -> Optional[str]:
        """Obtiene el timestamp de la última verificación de configuración"""
        try:
            # Esto se coordinaría con el ConfigSyncClient
            from sync.config_sync_client import get_config_sync_client
            client = get_config_sync_client()
            if hasattr(client, 'last_check_time') and client.last_check_time:
                return client.last_check_time.isoformat()
        except Exception:
            pass
        return None
    
    def _get_log_info(self) -> Optional[Dict[str, Any]]:
        """Obtiene información de archivos de log"""
        try:
            log_dir = Path('logs')
            if not log_dir.exists():
                return None
            
            log_files = list(log_dir.glob('*.log'))
            if not log_files:
                return None
            
            total_size = sum(f.stat().st_size for f in log_files)
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            return {
                'total_files': len(log_files),
                'total_size_mb': round(total_size / (1024*1024), 2),
                'latest_log': str(latest_log.name),
                'latest_modified': datetime.fromtimestamp(
                    latest_log.stat().st_mtime
                ).isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"Error obteniendo información de logs: {e}")
            return None
    
    def _process_heartbeat_response(self, response_data: Dict[str, Any]):
        """
        Procesa la respuesta del servidor al heartbeat.
        
        Args:
            response_data: Datos de respuesta del servidor
        """
        try:
            # Verificar si el servidor tiene instrucciones
            if response_data.get('config_pending'):
                self.logger.info("Servidor indica configuración pendiente")
                # Aquí podrías notificar al ConfigSyncClient para que fuerce una verificación
            
            # Procesar otros comandos del servidor si los hay
            commands = response_data.get('commands', [])
            for command in commands:
                self._process_server_command(command)
                
        except Exception as e:
            self.logger.warning(f"Error procesando respuesta de heartbeat: {e}")
    
    def _process_server_command(self, command: Dict[str, Any]):
        """
        Procesa un comando del servidor.
        
        Args:
            command: Comando a procesar
        """
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'force_config_sync':
                self.logger.info("Comando del servidor: forzar sincronización de configuración")
                # Aquí notificarías al ConfigSyncClient
                
            elif cmd_type == 'restart_detection':
                self.logger.info("Comando del servidor: reiniciar detección")
                # Aquí notificarías al sistema principal
                
            elif cmd_type == 'update_log_level':
                new_level = command.get('level', 'INFO')
                self.logger.info(f"Comando del servidor: cambiar nivel de log a {new_level}")
                # Aquí cambiarías el nivel de logging
                
            else:
                self.logger.warning(f"Comando desconocido del servidor: {cmd_type}")
                
        except Exception as e:
            self.logger.error(f"Error procesando comando del servidor: {e}")


# Instancia global del enviador de heartbeats
_heartbeat_sender = None

def get_heartbeat_sender() -> HeartbeatSender:
    """Obtiene la instancia global del enviador de heartbeats"""
    global _heartbeat_sender
    if _heartbeat_sender is None:
        _heartbeat_sender = HeartbeatSender()
    return _heartbeat_sender