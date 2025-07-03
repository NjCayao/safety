# sync/device_auth.py
"""
Sistema de autenticación para dispositivos Raspberry Pi
Maneja tokens JWT y comunicación segura con el servidor
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from config.config_manager import get_config

class DeviceAuthenticator:
    """
    Maneja la autenticación del dispositivo con el servidor.
    Gestiona tokens JWT, renovación automática y almacenamiento seguro.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('DeviceAuth')
        
        # Configuración desde config manager
        self.server_url = get_config('sync.server_url', 'http://localhost/safety_system')
        self.device_id = get_config('device.device_id', self._get_default_device_id())
        self.api_key = get_config('device.api_key', None)
        
        # Endpoints de autenticación
        self.auth_endpoint = f"{self.server_url}/api/v1/auth/authenticate"
        self.verify_endpoint = f"{self.server_url}/api/v1/auth/verify"
        
        # Almacenamiento de token
        self.token_file = Path('config/.device_token')
        self.token_file.parent.mkdir(exist_ok=True)
        
        # Estado interno
        self._current_token = None
        self._token_expires_at = None
        self._last_auth_attempt = None
        
        # Cargar token existente si es válido
        self._load_stored_token()
    
    def get_device_id(self) -> str:
        """Obtiene el ID del dispositivo"""
        return self.device_id
    
    def get_valid_token(self) -> Optional[str]:
        """
        Obtiene un token válido, renovándolo si es necesario.
        
        Returns:
            str: Token JWT válido o None si no se puede autenticar
        """
        # Verificar si tenemos token válido
        if self._is_token_valid():
            return self._current_token
        
        # Intentar renovar token
        if self._authenticate():
            return self._current_token
        
        self.logger.warning("No se pudo obtener token válido")
        return None
    
    def refresh_token(self) -> bool:
        """
        Fuerza la renovación del token.
        
        Returns:
            bool: True si se renovó exitosamente
        """
        self._current_token = None
        self._token_expires_at = None
        return self._authenticate()
    
    def is_authenticated(self) -> bool:
        """
        Verifica si el dispositivo está autenticado.
        
        Returns:
            bool: True si está autenticado
        """
        return self._is_token_valid()
    
    def _get_default_device_id(self) -> str:
        """
        Genera un device_id por defecto basado en el hardware.
        
        Returns:
            str: Device ID único
        """
        try:
            # En Raspberry Pi, usar el serial number
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('Serial'):
                            serial = line.split(':')[1].strip()
                            return f"RPI_{serial[-8:]}"  # Últimos 8 caracteres
            
            # Fallback: usar MAC address
            import uuid
            mac = hex(uuid.getnode())[2:].upper()
            return f"DEV_{mac[-8:]}"
            
        except Exception as e:
            self.logger.warning(f"No se pudo generar device_id automático: {e}")
            return "DEV_UNKNOWN"
    
    def _is_token_valid(self) -> bool:
        """
        Verifica si el token actual es válido.
        
        Returns:
            bool: True si el token es válido
        """
        if not self._current_token or not self._token_expires_at:
            return False
        
        # Verificar si no ha expirado (con margen de 5 minutos)
        now = datetime.now()
        expires_with_margin = self._token_expires_at - timedelta(minutes=5)
        
        return now < expires_with_margin
    
    def _authenticate(self) -> bool:
        """
        Realiza la autenticación con el servidor.
        
        Returns:
            bool: True si la autenticación fue exitosa
        """
        # Evitar intentos muy frecuentes
        if (self._last_auth_attempt and 
            datetime.now() - self._last_auth_attempt < timedelta(minutes=1)):
            return False
        
        self._last_auth_attempt = datetime.now()
        
        if not self.api_key:
            self.logger.error("API key no configurada")
            return False
        
        try:
            # Datos de autenticación
            auth_data = {
                'device_id': self.device_id,
                'api_key': self.api_key
            }
            
            # Headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'SafetySystem-Pi/{self.device_id}'
            }
            
            # Realizar solicitud de autenticación
            response = requests.post(
                self.auth_endpoint,
                json=auth_data,
                headers=headers,
                timeout=get_config('sync.connection_timeout', 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'success':
                    # Extraer token y tiempo de expiración
                    token_data = result.get('data', {})
                    self._current_token = token_data.get('token')
                    expires_in = token_data.get('expires_in', 3600)  # 1 hora por defecto
                    
                    self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    # Guardar token
                    self._save_token()
                    
                    self.logger.info(f"Autenticación exitosa. Token expira en {expires_in/3600:.1f} horas")
                    return True
                else:
                    self.logger.error(f"Error de autenticación: {result.get('message', 'Error desconocido')}")
                    return False
            else:
                self.logger.error(f"Error HTTP en autenticación: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de conexión en autenticación: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error inesperado en autenticación: {e}")
            return False
    
    def _save_token(self):
        """Guarda el token en archivo para persistencia"""
        if not self._current_token or not self._token_expires_at:
            return
        
        try:
            token_data = {
                'token': self._current_token,
                'expires_at': self._token_expires_at.isoformat(),
                'device_id': self.device_id,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            # Establecer permisos restrictivos (solo owner puede leer)
            os.chmod(self.token_file, 0o600)
            
        except Exception as e:
            self.logger.warning(f"No se pudo guardar token: {e}")
    
    def _load_stored_token(self):
        """Carga token guardado si es válido"""
        if not self.token_file.exists():
            return
        
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            # Verificar que sea para el mismo dispositivo
            if token_data.get('device_id') != self.device_id:
                self.logger.info("Token almacenado es para otro dispositivo, ignorando")
                return
            
            # Cargar datos del token
            self._current_token = token_data.get('token')
            expires_at_str = token_data.get('expires_at')
            
            if expires_at_str:
                self._token_expires_at = datetime.fromisoformat(expires_at_str)
                
                # Verificar si sigue siendo válido
                if self._is_token_valid():
                    self.logger.info("Token almacenado cargado y válido")
                else:
                    self.logger.info("Token almacenado ha expirado")
                    self._current_token = None
                    self._token_expires_at = None
            
        except Exception as e:
            self.logger.warning(f"Error cargando token almacenado: {e}")
            # Limpiar archivo corrupto
            try:
                self.token_file.unlink()
            except:
                pass
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Obtiene headers de autorización para requests.
        
        Returns:
            Dict con headers de autorización
        """
        token = self.get_valid_token()
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'SafetySystem-Pi/{self.device_id}'
        }
        
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        return headers
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba la conexión con el servidor.
        
        Returns:
            Dict con resultado de la prueba
        """
        try:
            # Verificar conectividad básica
            response = requests.get(
                f"{self.server_url}/api/v1/devices/status",
                headers=self.get_auth_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'message': 'Conexión exitosa con el servidor',
                    'server_time': result.get('timestamp'),
                    'device_status': result.get('device_status')
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'message': 'Error de autenticación - token inválido',
                    'needs_reauth': True
                }
            else:
                return {
                    'success': False,
                    'message': f'Error del servidor: {response.status_code}'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': 'No se puede conectar al servidor - verificar red'
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'message': 'Timeout de conexión - servidor no responde'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error inesperado: {e}'
            }
    
    def cleanup(self):
        """Limpia recursos y archivos temporales"""
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                self.logger.info("Token almacenado eliminado")
        except Exception as e:
            self.logger.warning(f"Error limpiando archivos: {e}")


# Instancia global del autenticador
_device_authenticator = None

def get_device_authenticator() -> DeviceAuthenticator:
    """
    Obtiene la instancia global del autenticador.
    Patrón Singleton para evitar múltiples instancias.
    """
    global _device_authenticator
    if _device_authenticator is None:
        _device_authenticator = DeviceAuthenticator()
    return _device_authenticator


# Funciones de conveniencia
def get_auth_headers() -> Dict[str, str]:
    """Función de conveniencia para obtener headers de auth"""
    return get_device_authenticator().get_auth_headers()

def get_device_id() -> str:
    """Función de conveniencia para obtener device ID"""
    return get_device_authenticator().get_device_id()

def is_authenticated() -> bool:
    """Función de conveniencia para verificar autenticación"""
    return get_device_authenticator().is_authenticated()

def test_server_connection() -> Dict[str, Any]:
    """Función de conveniencia para probar conexión"""
    return get_device_authenticator().test_connection()