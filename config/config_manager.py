# config/config_manager.py
import os
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

class ConfigManager:
    """
    Sistema de configuración centralizado y jerárquico.
    Carga configuración en este orden de prioridad:
    1. Configuración específica de entorno (development.yaml / production.yaml)
    2. Configuración por defecto (default.yaml)
    3. Valores hardcodeados como fallback
    """
    
    def __init__(self, config_dir: str = None):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_dir: Directorio de configuración. Si es None, usa ./config/
        """
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger('ConfigManager')
        
        # Detectar entorno automáticamente
        self.environment = self._detect_environment()
        
        # Cargar configuración
        self.config = {}
        self._load_all_configs()
        
        self.logger.info(f"ConfigManager inicializado - Entorno: {self.environment}")
    
    def _detect_environment(self) -> str:
        """
        Detecta automáticamente el entorno de ejecución.
        
        Returns:
            str: 'development' o 'production'
        """
        # Variable de entorno explícita
        env = os.environ.get('SAFETY_SYSTEM_ENV')
        if env:
            return env.lower()
        
        # Detectar por sistema operativo y disponibilidad de GUI
        if os.name == 'nt':  # Windows
            return 'development'
        
        # Linux - verificar si tiene GUI disponible
        if os.environ.get('DISPLAY') is None:
            return 'production'  # Sin GUI = Raspberry Pi
        else:
            return 'development'  # Con GUI = desarrollo en Linux
    
    def _load_all_configs(self):
        """Carga todos los archivos de configuración en orden jerárquico."""
        
        # 1. Configuración por defecto
        self._load_config_file('default.yaml')
        
        # 2. Configuración específica de entorno
        env_file = f'{self.environment}.yaml'
        self._load_config_file(env_file)
        
        # 3. Configuración local (si existe) - no versionada
        self._load_config_file('local.yaml', required=False)
        
        self.logger.info(f"Configuración cargada para entorno: {self.environment}")
    
    def _load_config_file(self, filename: str, required: bool = True):
        """
        Carga un archivo de configuración específico.
        
        Args:
            filename: Nombre del archivo YAML
            required: Si True, registra error si no existe
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            if required:
                self.logger.warning(f"Archivo de configuración no encontrado: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
                self._merge_config(self.config, file_config)
                self.logger.debug(f"Configuración cargada de: {filename}")
        except Exception as e:
            self.logger.error(f"Error cargando configuración de {filename}: {e}")
    
    def _merge_config(self, base: Dict, update: Dict):
        """
        Fusiona configuración de manera recursiva.
        
        Args:
            base: Configuración base (se modifica in-place)
            update: Configuración a fusionar
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Args:
            key: Clave en formato 'seccion.subseccion.valor'
            default: Valor por defecto si no se encuentra la clave
            
        Returns:
            Valor de configuración o default
        
        Examples:
            config.get('fatigue.eye_closed_threshold', 1.5)
            config.get('camera.fps', 30)
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                self.logger.debug(f"Configuración '{key}' no encontrada, usando default: {default}")
            return default
    
    def set(self, key: str, value: Any):
        """
        Establece un valor de configuración en runtime.
        
        Args:
            key: Clave en formato 'seccion.subseccion.valor'
            value: Valor a establecer
        """
        keys = key.split('.')
        config = self.config
        
        # Navegar hasta el penúltimo nivel
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Establecer el valor final
        config[keys[-1]] = value
        self.logger.debug(f"Configuración '{key}' establecida a: {value}")
    
    def get_section(self, section: str) -> Dict:
        """
        Obtiene una sección completa de configuración.
        
        Args:
            section: Nombre de la sección
            
        Returns:
            Diccionario con toda la sección
        """
        return self.config.get(section, {})
    
    def is_development(self) -> bool:
        """Retorna True si estamos en entorno de desarrollo."""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Retorna True si estamos en entorno de producción."""
        return self.environment == 'production'
    
    def has_gui(self) -> bool:
        """Retorna True si el entorno soporta GUI."""
        return self.get('system.enable_gui', self.is_development())
    
    def get_log_level(self) -> str:
        """Obtiene el nivel de logging configurado."""
        return self.get('system.log_level', 'INFO' if self.is_production() else 'DEBUG')
    
    def validate_config(self) -> bool:
        """
        Valida que la configuración tenga valores válidos.
        
        Returns:
            True si la configuración es válida
        """
        try:
            # Validaciones básicas
            fps = self.get('camera.fps', 30)
            if not isinstance(fps, (int, float)) or fps <= 0 or fps > 60:
                self.logger.error(f"FPS inválido: {fps}")
                return False
            
            # Validar umbrales de fatiga
            ear_threshold = self.get('fatigue.eye_closed_threshold', 1.5)
            if not isinstance(ear_threshold, (int, float)) or ear_threshold <= 0:
                self.logger.error(f"Umbral EAR inválido: {ear_threshold}")
                return False
            
            # Agregar más validaciones según necesidad
            
            self.logger.info("Configuración validada correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando configuración: {e}")
            return False
    
    def reload(self):
        """Recarga la configuración desde archivos."""
        self.config.clear()
        self._load_all_configs()
        self.logger.info("Configuración recargada")
    
    def dump_config(self) -> str:
        """
        Retorna la configuración actual como string YAML.
        Útil para debugging.
        """
        return yaml.dump(self.config, default_flow_style=False, indent=2)


# Instancia global del gestor de configuración
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Obtiene la instancia global del gestor de configuración.
    Patrón Singleton para evitar múltiples cargas.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Funciones de conveniencia
def get_config(key: str, default: Any = None) -> Any:
    """Función de conveniencia para obtener configuración."""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any):
    """Función de conveniencia para establecer configuración."""
    get_config_manager().set(key, value)

def is_development() -> bool:
    """Función de conveniencia para verificar entorno de desarrollo."""
    return get_config_manager().is_development()

def is_production() -> bool:
    """Función de conveniencia para verificar entorno de producción."""
    return get_config_manager().is_production()

def has_gui() -> bool:
    """Función de conveniencia para verificar disponibilidad de GUI."""
    return get_config_manager().has_gui()