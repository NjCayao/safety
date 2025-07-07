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

# Importar configuración si está disponible
try:
    from config.config_manager import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

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
            self.auto_send = get_config('reports.auto_send', True)
            self.save_images = get_config('reports.save_images', True)
            self.retention_days = get_config('reports.retention_days', 30)
        else:
            self.server_url = server_url or 'http://localhost:5000'
            self.auto_send = True
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
        
        # Iniciar hilo de envío si está habilitado
        if self.auto_send:
            self._start_sender_thread()
    
    def _create_directories(self):
        """Crea la estructura de directorios para reportes"""
        modules = ['fatigue', 'behavior', 'distraction', 'yawn', 'general']
        
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
            
            # Crear estructura de reporte
            report = {
                'id': report_id,
                'module': module_name,
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'operator': operator_info,
                'data': data,
                'metadata': {
                    'version': '1.0',
                    'generated_by': 'ReportManager',
                    'environment': os.environ.get('ENVIRONMENT', 'production')
                }
            }
            
            # Paths para archivos
            date_path = timestamp.strftime("%Y/%m")
            base_dir = os.path.join(self.reports_dir, module_name, date_path)
            os.makedirs(base_dir, exist_ok=True)
            
            base_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{event_type}_{operator_info.get('id', 'unknown')}"
            
            # Guardar imagen si se proporciona
            if frame is not None and self.save_images:
                img_path = os.path.join(base_dir, f"{base_filename}.jpg")
                success = cv2.imwrite(img_path, frame)
                if success:
                    report['image_path'] = img_path
                    self.logger.debug(f"Imagen guardada: {img_path}")
            
            # Guardar JSON del reporte
            json_path = os.path.join(base_dir, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            report['json_path'] = json_path
            
            # Actualizar estadísticas
            self.stats['reports_generated'] += 1
            self.stats['last_report_time'] = timestamp
            
            self.logger.info(f"Reporte generado: {report_id}")
            
            # Agregar a cola de envío si está habilitado
            if self.auto_send:
                self.report_queue.put(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return None
    
    def _generate_report_id(self, module, event_type, timestamp, operator_info):
        """Genera un ID único para el reporte"""
        operator_id = operator_info.get('id', 'unknown') if operator_info else 'unknown'
        return f"{module}_{event_type}_{operator_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    
    def send_report(self, report):
        """
        Envía un reporte al servidor.
        
        Args:
            report: Diccionario con el reporte
            
        Returns:
            bool: True si se envió correctamente
        """
        try:
            import requests
            
            # Preparar datos para envío
            send_data = {
                'id': report['id'],
                'module': report['module'],
                'event_type': report['event_type'],
                'timestamp': report['timestamp'],
                'operator': report['operator'],
                'data': report['data']
            }
            
            # Enviar JSON
            response = requests.post(
                f"{self.server_url}/api/reports",
                json=send_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.stats['reports_sent'] += 1
                self.logger.info(f"Reporte enviado: {report['id']}")
                
                # Si hay imagen, enviarla también
                if 'image_path' in report and os.path.exists(report['image_path']):
                    self._send_image(report['id'], report['image_path'])
                
                return True
            else:
                self.logger.error(f"Error enviando reporte: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enviando reporte: {e}")
            self.stats['reports_failed'] += 1
            return False
    
    def _send_image(self, report_id, image_path):
        """Envía imagen asociada al reporte"""
        try:
            import requests
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'report_id': report_id}
                
                response = requests.post(
                    f"{self.server_url}/api/reports/image",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    self.logger.debug(f"Imagen enviada para reporte: {report_id}")
                else:
                    self.logger.error(f"Error enviando imagen: {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Error enviando imagen: {e}")
    
    def _start_sender_thread(self):
        """Inicia el hilo para envío asíncrono de reportes"""
        self.running = True
        self.sender_thread = threading.Thread(target=self._sender_worker)
        self.sender_thread.daemon = True
        self.sender_thread.start()
        self.logger.info("Hilo de envío de reportes iniciado")
    
    def _sender_worker(self):
        """Worker para enviar reportes de forma asíncrona"""
        while self.running:
            try:
                # Esperar por reportes con timeout
                report = self.report_queue.get(timeout=1)
                
                # Intentar enviar
                success = self.send_report(report)
                
                if not success:
                    # Reintentar una vez después de esperar
                    time.sleep(2)
                    self.send_report(report)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en sender worker: {e}")
    
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
                
                # Recorrer años y meses
                for year in os.listdir(module_path):
                    year_path = os.path.join(module_path, year)
                    if not os.path.isdir(year_path):
                        continue
                    
                    for month in os.listdir(year_path):
                        month_path = os.path.join(year_path, month)
                        if not os.path.isdir(month_path):
                            continue
                        
                        # Verificar archivos
                        for filename in os.listdir(month_path):
                            file_path = os.path.join(month_path, filename)
                            
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
            'queue_size': self.report_queue.qsize() if self.report_queue else 0,
            'sender_active': self.sender_thread.is_alive() if self.sender_thread else False
        }
    
    def stop(self):
        """Detiene el gestor de reportes"""
        self.running = False
        
        if self.sender_thread:
            self.sender_thread.join(timeout=5)
        
        # Procesar reportes pendientes
        while not self.report_queue.empty():
            try:
                report = self.report_queue.get_nowait()
                self.send_report(report)
            except:
                break
        
        self.logger.info("Gestor de reportes detenido")

# Singleton para uso global
_report_manager = None

def get_report_manager():
    """Obtiene la instancia global del gestor de reportes"""
    global _report_manager
    if _report_manager is None:
        _report_manager = ReportManager()
    return _report_manager