"""
Report Manager
==============
Gestiona la generación y envío de reportes para todos los módulos.
"""

import os
import json
import cv2
from datetime import datetime
import logging
import threading
import queue
import time
import numpy as np

# Importar configuración si está disponible
try:
    from config.config_manager import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos numpy"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.void):
            return None
        return super(NumpyEncoder, self).default(obj)

class ReportManager:
    def __init__(self, reports_dir="reports", server_url=None):
        """
        Inicializa el gestor de reportes.
        
        Args:
            reports_dir: Directorio base para guardar reportes
            server_url: URL del servidor para envío de reportes
        """
        self.reports_dir = reports_dir
        self.logger = logging.getLogger('ReportManager')
        
        # Configuración
        if CONFIG_AVAILABLE:
            self.server_url = server_url or get_config('server.url', 'http://localhost:5000')
            self.auto_send = get_config('reports.auto_send', False)  # CAMBIO: False por defecto
            self.save_images = get_config('reports.save_images', True)
            self.retention_days = get_config('reports.retention_days', 30)
        else:
            self.server_url = server_url or 'http://localhost:5000'
            self.auto_send = False  # CAMBIO: Deshabilitado por defecto
            self.save_images = True
            self.retention_days = 30
        
        # Crear estructura de directorios
        self._create_directories()
        
        # Cola para envío asíncrono
        self.report_queue = queue.Queue()
        self.sender_thread = None
        self.running = False
        
        # Estadísticas
        self.stats = {
            'reports_generated': 0,
            'reports_sent': 0,
            'reports_failed': 0,
            'last_report_time': None
        }
        
        # NO iniciar hilo de envío automáticamente
        if self.auto_send:
            self.logger.info("Envío automático deshabilitado - Los reportes se guardarán localmente")
    
    def _create_directories(self):
        """Crea la estructura de directorios para reportes"""
        modules = ['fatigue', 'behavior', 'distraction', 'yawn', 'analysis', 'general', 'face_recognition']        
        
        for module in modules:
            module_dir = os.path.join(self.reports_dir, module)
            os.makedirs(module_dir, exist_ok=True)
        
        self.logger.info(f"Estructura de reportes creada en: {self.reports_dir}")
    
    def generate_report(self, module_name, event_type, data, frame=None, operator_info=None):
        """
        Genera un reporte para un evento específico.
        
        Args:
            module_name: Nombre del módulo (fatigue, behavior, etc.)
            event_type: Tipo de evento (microsleep, phone_use, etc.)
            data: Datos del evento
            frame: Frame de video (opcional)
            operator_info: Información del operador
            
        Returns:
            dict: Reporte generado con paths
        """
        try:
            timestamp = datetime.now()
            
            # Crear identificador único
            report_id = self._generate_report_id(module_name, event_type, timestamp, operator_info)
            
            # Función para convertir tipos numpy
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif isinstance(obj, set):
                    return list(obj)  # Convertir sets a listas
                return obj
            
            # Convertir datos antes de crear el reporte
            clean_data = convert_numpy_types(data)
            clean_operator_info = convert_numpy_types(operator_info) if operator_info else None
            
            # Crear estructura de reporte
            report = {
                'id': report_id,
                'module': module_name,
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'operator': clean_operator_info,
                'data': clean_data,
                'metadata': {
                    'version': '1.0',
                    'generated_by': 'ReportManager',
                    'environment': os.environ.get('ENVIRONMENT', 'production')
                }
            }
            
            # Paths para archivos
            date_path = timestamp.strftime("%Y/%m/%d")  # Agregar día también
            base_dir = os.path.join(self.reports_dir, module_name, date_path)
            os.makedirs(base_dir, exist_ok=True)
            
            base_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{event_type}_{operator_info.get('id', 'unknown') if operator_info else 'unknown'}"
            
            # Guardar imagen si se proporciona
            if frame is not None and self.save_images:
                img_path = os.path.join(base_dir, f"{base_filename}.jpg")
                success = cv2.imwrite(img_path, frame)
                if success:
                    report['image_path'] = img_path
                    self.logger.debug(f"Imagen guardada: {img_path}")
                else:
                    self.logger.error(f"Error guardando imagen: {img_path}")
            
            # Guardar JSON del reporte
            json_path = os.path.join(base_dir, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            report['json_path'] = json_path
            
            # Actualizar estadísticas
            self.stats['reports_generated'] += 1
            self.stats['last_report_time'] = timestamp
            
            self.logger.info(f"Reporte generado y guardado localmente: {report_id}")
            
            # NO agregar a cola de envío
            # El servidor recogerá los archivos directamente
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_report_id(self, module, event_type, timestamp, operator_info):
        """Genera un ID único para el reporte"""
        operator_id = operator_info.get('id', 'unknown') if operator_info else 'unknown'
        return f"{module}_{event_type}_{operator_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    
    def send_report(self, report):
        """
        Envía un reporte al servidor (solo si se llama explícitamente).
        
        Args:
            report: Diccionario con el reporte
            
        Returns:
            bool: True si se envió correctamente
        """
        # Esta función queda disponible pero no se usará automáticamente
        self.logger.warning("Envío de reportes deshabilitado - Los reportes se guardan localmente")
        return False
    
    def _start_sender_thread(self):
        """NO iniciar hilo de envío"""
        self.logger.info("Hilo de envío deshabilitado - Los reportes se guardarán solo localmente")
    
    def cleanup_old_reports(self):
        """Limpia reportes antiguos según política de retención"""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cleaned_count = 0
            
            for module in os.listdir(self.reports_dir):
                module_path = os.path.join(self.reports_dir, module)
                if not os.path.isdir(module_path):
                    continue
                
                # Recorrer estructura de directorios
                for root, dirs, files in os.walk(module_path):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        
                        # Obtener fecha de modificación
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            cleaned_count += 1
            
            self.logger.info(f"Limpieza completada: {cleaned_count} archivos eliminados")
            
        except Exception as e:
            self.logger.error(f"Error en limpieza: {e}")
    
    def get_statistics(self):
        """Obtiene estadísticas del gestor de reportes"""
        return {
            **self.stats,
            'queue_size': 0,  # Sin cola de envío
            'sender_active': False  # Sin hilo de envío
        }
    
    def stop(self):
        """Detiene el gestor de reportes"""
        self.running = False
        self.logger.info("Gestor de reportes detenido")

# Singleton para uso global
_report_manager = None

def get_report_manager():
    """Obtiene la instancia global del gestor de reportes"""
    global _report_manager
    if _report_manager is None:
        _report_manager = ReportManager()
    return _report_manager