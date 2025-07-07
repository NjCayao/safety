import os
import cv2
from datetime import datetime

class ReportGenerator:
    def __init__(self, reports_dir):
        """Inicializa el generador de reportes"""
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
    
    def generate(self, frame, event_type, operator_info=None):
        """Genera un reporte con imagen y metadatos"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            operator_id = operator_info["id"] if operator_info else "unknown"
            filename = f"{operator_id}_{event_type}_{timestamp}"
            
            # Guardar imagen
            img_path = os.path.join(self.reports_dir, f"{filename}.jpg")
            cv2.imwrite(img_path, frame)
            
            # Generar reporte de texto
            self._generate_text_report(filename, event_type, operator_info)
            
            return True
        except Exception as e:
            print(f"Error generating report: {e}")
            return False
    
    def _generate_text_report(self, filename, event_type, operator_info):
        """Genera el archivo de texto con detalles del evento"""
        txt_path = os.path.join(self.reports_dir, f"{filename}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"=== Reporte de Seguridad ===\n")
            f.write(f"Tipo de evento: {event_type}\n")
            f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if operator_info:
                f.write("\n=== Información del Operador ===\n")
                f.write(f"ID: {operator_info.get('id', 'N/A')}\n")
                f.write(f"Nombre: {operator_info.get('name', 'N/A')}\n")
            
            f.write("\n=== Acciones Recomendadas ===\n")
            if "fatigue" in event_type:
                f.write("- Tomar un descanso inmediato de 15-20 minutos\n")
                f.write("- Realizar ejercicios de estiramiento\n")
            elif "yawn" in event_type:
                f.write("- Realizar pausas activas\n")
                f.write("- Hidratarse adecuadamente\n")
            elif "distraction" in event_type:
                f.write("- Reenfocar la atención en la tarea\n")
                f.write("- Verificar condiciones del entorno de trabajo\n")