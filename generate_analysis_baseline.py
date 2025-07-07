"""
Script para generar analysis_baseline.json de un operador específico
"""
import sys
import os

# Agregar rutas necesarias
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.master_calibration_manager import MasterCalibrationManager

def generate_baseline_for_operator(operator_id):
    """Genera baseline solo para un operador específico"""
    
    # Crear instancia del calibrador maestro
    manager = MasterCalibrationManager()
    
    # Ruta a las fotos del operador
    photos_path = f"server/operator-photo/{operator_id}"
    
    # Verificar que existe
    if not os.path.exists(photos_path):
        print(f"❌ No se encontró el directorio: {photos_path}")
        return False
    
    # Calibrar solo este operador
    print(f"🔄 Generando baselines para operador {operator_id}...")
    
    # Leer nombre del operador
    info_file = os.path.join(photos_path, "info.txt")
    operator_name = "Desconocido"
    
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if line.startswith("Operador:"):
                    operator_name = line.replace("Operador:", "").strip()
        except:
            pass
    
    # Ejecutar calibración
    success = manager.calibrate_operator(operator_id, photos_path, operator_name)
    
    if success:
        print(f"✅ Baselines generados exitosamente para {operator_name} ({operator_id})")
        print(f"📁 Archivos creados en: operators/baseline-json/{operator_id}/")
        
        # Verificar que se creó analysis_baseline.json
        analysis_path = f"operators/baseline-json/{operator_id}/analysis_baseline.json"
        if os.path.exists(analysis_path):
            print(f"✅ analysis_baseline.json creado correctamente")
        else:
            print(f"⚠️  analysis_baseline.json NO se creó (necesitas agregar AnalysisCalibration)")
    else:
        print(f"❌ Error generando baselines")
    
    return success


if __name__ == "__main__":
    # Operador específico
    OPERATOR_ID = "47469940"
    
    # Generar baseline
    generate_baseline_for_operator(OPERATOR_ID)