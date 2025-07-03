import logging
import logging.config
import os
from datetime import datetime

def setup_logging():
    """Configurar sistema de logging"""
    # Asegurar que existe el directorio de logs
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Nombre de archivo con fecha
    log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Configuración
    config = {
        'version': 1,
        'formatters': {
            'verbose': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'simple': {
                'format': '%(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'verbose',
                'filename': log_file,
                'encoding': 'utf8'
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
            },
            'api_client': {
                'level': 'DEBUG',
                'propagate': True,
            },
            'local_storage': {
                'level': 'DEBUG',
                'propagate': True,
            },
            'connection': {
                'level': 'INFO',
                'propagate': True,
            },
            'sync': {
                'level': 'DEBUG',
                'propagate': True,
            },
            'file_manager': {
                'level': 'INFO',
                'propagate': True,
            },
        }
    }
    
    # Aplicar configuración
    logging.config.dictConfig(config)
    
    return logging.getLogger('main')