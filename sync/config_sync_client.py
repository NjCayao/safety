# sync/config_sync_client.py
"""
Cliente de sincronización de configuración para Raspberry Pi
Consulta automáticamente por cambios y los aplica sin interrumpir el sistema
"""

import time
import json
import logging
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from config.config_manager import get_config_manager, get_config
from sync.device_auth import get_device_authenticator

class ConfigSyncClient:
    """
    Cliente que sincroniza configuración desde el servidor web.
    Se ejecuta en un hilo separado y consulta periódicamente por cambios.
    """
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.authenticator = get_device_authenticator()
        self.logger = logging.getLogger('ConfigSyncClient')
        
        # Configuración de sincronización
        self.server_url = get_config('sync.server_url', 'http://localhost/safety_system')
        self.check_interval = get_config('sync.config_check_interval', 60)  # 1 minuto
        self.connection_timeout = get_config('sync.connection_timeout', 30)
        
        # Estado interno
        self.last_config_version = 1
        self.last_check_time = None
        self.is_running = False
        self._sync_thread = None
        self._stop_event = threading.Event()
        
        # API endpoints
        self.config_endpoint = f"{self.server_url}/api/v1/devices/config"
        
        # Callbacks para notificar cambios al sistema principal
        self._config_change_callbacks = []
        
        self.logger.info("Cliente de sincronización inicializado")
    
    def add_config_change_callback(self, callback):
        """
        Agrega callback que se ejecuta cuando cambia la configuración.
        
        Args:
            callback: Función que recibe (old_config, new_config)
        """
        self._config_change_callbacks.append(callback)
    
    def start(self):
        """Inicia el cliente de sincronización en un hilo separado"""
        if self.is_running:
            self.logger.warning("Cliente ya está ejecutándose")
            return
        
        self.logger.info("Iniciando cliente de sincronización de configuración")
        self.is_running = True
        self._stop_event.clear()
        
        # Crear y iniciar hilo
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            name="ConfigSyncClient",
            daemon=True
        )
        self._sync_thread.start()
    
    def stop(self):
        """Detiene el cliente de sincronización"""
        if not self.is_running:
            return
        
        self.logger.info("Deteniendo cliente de sincronización")
        self.is_running = False
        self._stop_event.set()
        
        # Esperar a que termine el hilo
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
    
    def force_sync(self) -> bool:
        """
        Fuerza una sincronización inmediata.
        
        Returns:
            bool: True si se aplicaron cambios
        """
        self.logger.info("Forzando sincronización de configuración")
        return self._check_for_config_updates()
    
    def _sync_loop(self):
        """Bucle principal de sincronización"""
        self.logger.info("Bucle de sincronización iniciado")
        
        while self.is_running and not self._stop_event.is_set():
            try:
                # Verificar configuración
                config_updated = self._check_for_config_updates()
                
                if config_updated:
                    self.logger.info("Configuración actualizada exitosamente")
                
                # Esperar antes del próximo check
                if self._stop_event.wait(self.check_interval):
                    break  # Se solicitó parar
                
            except Exception as e:
                self.logger.error(f"Error en bucle de sincronización: {e}")
                # Esperar más tiempo en caso de error
                if self._stop_event.wait(min(self.check_interval * 2, 300)):
                    break
        
        self.logger.info("Bucle de sincronización terminado")
    
    def _check_for_config_updates(self) -> bool:
        """
        Consulta al servidor si hay configuración pendiente.
        
        Returns:
            bool: True si se aplicó una nueva configuración
        """
        try:
            # Obtener headers de autenticación
            headers = self.authenticator.get_auth_headers()
            
            if 'Authorization' not in headers:
                self.logger.warning("No hay token de autenticación válido")
                return False
            
            # Parámetros de consulta
            params = {
                'device_id': self.authenticator.get_device_id(),
                'current_version': self.last_config_version
            }
            
            # Consultar configuración
            response = requests.get(
                self.config_endpoint,
                headers=headers,
                params=params,
                timeout=self.connection_timeout
            )
            
            if response.status_code == 200:
                config_data = response.json()
                return self._process_config_response(config_data)
            elif response.status_code == 401:
                self.logger.warning("Token expirado, renovando...")
                self.authenticator.refresh_token()
                return False
            elif response.status_code == 404:
                self.logger.warning("Dispositivo no encontrado en servidor")
                return False
            else:
                self.logger.warning(f"Error al consultar configuración: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Error de conexión al consultar configuración: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error inesperado al consultar configuración: {e}")
            return False
    
    def _process_config_response(self, config_data: Dict[str, Any]) -> bool:
        """
        Procesa la respuesta del servidor con configuración.
        
        Args:
            config_data: Datos de configuración del servidor
            
        Returns:
            bool: True si se aplicó configuración nueva
        """
        try:
            server_version = config_data.get('config_version', 1)
            config_pending = config_data.get('config_pending', False)
            new_config = config_data.get('config', {})
            
            # Verificar si hay cambios pendientes
            if not config_pending or server_version <= self.last_config_version:
                self.logger.debug("No hay configuración pendiente")
                return False
            
            self.logger.info(f"Nueva configuración disponible: v{server_version}")
            
            # Obtener configuración actual para backup
            old_config = self._get_current_config_snapshot()
            
            # Aplicar nueva configuración
            success = self._apply_new_config(new_config, server_version)
            
            if success:
                # Confirmar aplicación al servidor
                self._confirm_config_applied(server_version)
                self.last_config_version = server_version
                
                # Notificar callbacks
                self._notify_config_change(old_config, new_config)
                
                return True
            else:
                # Reportar error al servidor
                self._report_config_error("Error al aplicar configuración", server_version)
                return False
                
        except Exception as e:
            self.logger.error(f"Error procesando respuesta de configuración: {e}")
            self._report_config_error(f"Error procesando configuración: {e}")
            return False
    
    def _apply_new_config(self, new_config: Dict[str, Any], version: int) -> bool:
        """
        Aplica nueva configuración al sistema.
        
        Args:
            new_config: Nueva configuración a aplicar
            version: Versión de la configuración
            
        Returns:
            bool: True si se aplicó exitosamente
        """
        try:
            self.logger.info("Aplicando nueva configuración...")
            
            # Validar configuración antes de aplicarla
            if not self._validate_config(new_config):
                self.logger.error("Configuración inválida, abortando aplicación")
                return False
            
            # Hacer backup de configuración actual
            backup_success = self._backup_current_config()
            if not backup_success:
                self.logger.warning("No se pudo hacer backup de configuración actual")
            
            # Aplicar cada sección de configuración
            applied_sections = []
            try:
                for section, values in new_config.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            config_key = f"{section}.{key}"
                            old_value = self.config_manager.get(config_key)
                            
                            # Solo aplicar si el valor cambió
                            if old_value != value:
                                self.config_manager.set(config_key, value)
                                self.logger.debug(f"Configurado {config_key}: {old_value} → {value}")
                        
                        applied_sections.append(section)
                
                # Guardar configuración actualizada en archivos
                self._save_config_to_files(new_config)
                
                self.logger.info(f"Configuración v{version} aplicada exitosamente")
                return True
                
            except Exception as e:
                self.logger.error(f"Error aplicando configuración: {e}")
                
                # Intentar rollback parcial
                if backup_success:
                    self._restore_config_backup()
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error general aplicando configuración: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida que la configuración recibida sea válida.
        
        Args:
            config: Configuración a validar
            
        Returns:
            bool: True si es válida
        """
        try:
            # Verificar que sea un diccionario válido
            if not isinstance(config, dict):
                return False
            
            # Verificar secciones críticas
            required_sections = ['camera', 'system']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Sección requerida '{section}' no encontrada")
                    return False
            
            # Validar rangos de valores críticos
            validations = [
                ('camera.fps', 1, 60),
                ('camera.width', 160, 1920),
                ('camera.height', 120, 1080),
            ]
            
            for key, min_val, max_val in validations:
                section, param = key.split('.')
                if section in config and param in config[section]:
                    value = config[section][param]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        self.logger.error(f"Valor inválido para {key}: {value}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando configuración: {e}")
            return False
    
    def _get_current_config_snapshot(self) -> Dict[str, Any]:
        """Obtiene snapshot de la configuración actual"""
        try:
            return {
                'camera': self.config_manager.get_section('camera'),
                'fatigue': self.config_manager.get_section('fatigue'),
                'yawn': self.config_manager.get_section('yawn'),
                'distraction': self.config_manager.get_section('distraction'),
                'behavior': self.config_manager.get_section('behavior'),
                'audio': self.config_manager.get_section('audio'),
                'system': self.config_manager.get_section('system'),
                'sync': self.config_manager.get_section('sync'),
            }
        except Exception as e:
            self.logger.warning(f"Error obteniendo snapshot de configuración: {e}")
            return {}
    
    def _backup_current_config(self) -> bool:
        """Hace backup de la configuración actual"""
        try:
            import shutil
            
            config_dir = Path('config')
            backup_dir = Path('config/backup')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Hacer backup de archivos de configuración
            for config_file in ['production.yaml', 'development.yaml', 'local.yaml']:
                source = config_dir / config_file
                if source.exists():
                    backup_file = backup_dir / f"{config_file}.backup_{timestamp}"
                    shutil.copy2(source, backup_file)
            
            self.logger.debug(f"Backup de configuración creado: {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando backup de configuración: {e}")
            return False
    
    def _save_config_to_files(self, config: Dict[str, Any]):
        """Guarda la configuración en archivos YAML"""
        try:
            import yaml
            
            # Determinar qué archivo actualizar según el entorno
            if self.config_manager.is_production():
                config_file = Path('config/production.yaml')
            else:
                config_file = Path('config/development.yaml')
            
            # Leer configuración existente
            existing_config = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = yaml.safe_load(f) or {}
            
            # Mergear con nueva configuración
            for section, values in config.items():
                if section not in existing_config:
                    existing_config[section] = {}
                if isinstance(values, dict):
                    existing_config[section].update(values)
                else:
                    existing_config[section] = values
            
            # Guardar archivo actualizado
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)
            
            self.logger.debug(f"Configuración guardada en {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error guardando configuración en archivos: {e}")
    
    def _notify_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notifica callbacks sobre cambio de configuración"""
        try:
            for callback in self._config_change_callbacks:
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    self.logger.error(f"Error en callback de configuración: {e}")
        except Exception as e:
            self.logger.error(f"Error notificando cambio de configuración: {e}")
    
    def _confirm_config_applied(self, version: int):
        """Confirma al servidor que la configuración fue aplicada"""
        try:
            headers = self.authenticator.get_auth_headers()
            
            data = {
                'action': 'config_applied',
                'device_id': self.authenticator.get_device_id(),
                'config_version': version,
                'applied_at': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.config_endpoint,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Confirmación de configuración v{version} enviada")
            else:
                self.logger.warning(f"Error enviando confirmación: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error enviando confirmación de configuración: {e}")
    
    def _report_config_error(self, error_message: str, version: int = None):
        """Reporta error de configuración al servidor"""
        try:
            headers = self.authenticator.get_auth_headers()
            
            data = {
                'action': 'config_error',
                'device_id': self.authenticator.get_device_id(),
                'error_message': error_message,
                'error_time': datetime.now().isoformat()
            }
            
            if version:
                data['config_version'] = version
            
            response = requests.post(
                self.config_endpoint,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Error de configuración reportado al servidor")
            
        except Exception as e:
            self.logger.debug(f"Error reportando error de configuración: {e}")
    
    def _restore_config_backup(self):
        """Restaura configuración desde backup más reciente"""
        try:
            backup_dir = Path('config/backup')
            if not backup_dir.exists():
                return
            
            # Buscar backup más reciente
            backup_files = list(backup_dir.glob('*.backup_*'))
            if not backup_files:
                return
            
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            # Restaurar archivo
            original_name = latest_backup.name.split('.backup_')[0]
            original_path = Path('config') / original_name
            
            import shutil
            shutil.copy2(latest_backup, original_path)
            
            # Recargar configuración
            self.config_manager.reload()
            
            self.logger.info(f"Configuración restaurada desde backup: {latest_backup}")
            
        except Exception as e:
            self.logger.error(f"Error restaurando backup de configuración: {e}")


# Instancia global del cliente
_config_sync_client = None

def get_config_sync_client() -> ConfigSyncClient:
    """Obtiene la instancia global del cliente de sincronización"""
    global _config_sync_client
    if _config_sync_client is None:
        _config_sync_client = ConfigSyncClient()
    return _config_sync_client