#!/usr/bin/env python
"""
Calibración individual de operador
==================================
Ejecuta la calibración biométrica para un operador específico.
"""
import os
import sys
import time
from datetime import datetime

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.master_calibration_manager import MasterCalibrationManager

def update_progress(progress):
    """Actualiza el archivo de progreso con el porcentaje actual"""
    progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_progress.txt")
    with open(progress_file, "w") as f:
        f.write(str(progress))

def main():
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("ERROR: Debe proporcionar el ID del operador")
        sys.exit(1)
    
    operator_id = sys.argv[1]
    
    # Archivo de log
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_log.txt")
    
    # Guardar el directorio original ANTES de cualquier try
    original_dir = os.getcwd()
    
    # Inicializar progreso
    update_progress(0)
    
    try:
        # Log inicio
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando calibración para operador {operator_id}\n")
        
        update_progress(10)
        
        # Obtener rutas
        script_dir = os.path.dirname(os.path.abspath(__file__))  # /operators/
        base_dir = os.path.dirname(script_dir)  # /safety_system/
        
        # Cambiar al directorio base para que las rutas relativas funcionen
        os.chdir(base_dir)
        
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directorio de trabajo cambiado a: {os.getcwd()}\n")
        
        # Inicializar el manager con la ruta absoluta del modelo
        model_path = os.path.join(base_dir, "assets", "models", "shape_predictor_68_face_landmarks.dat")
        
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Modelo esperado en: {model_path}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ¿Existe el modelo? {os.path.exists(model_path)}\n")
        
        if not os.path.exists(model_path):
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No se encuentra el modelo\n")
            print("ERROR: No se encuentra el modelo")
            update_progress(100)
            os.chdir(original_dir)
            sys.exit(1)
        
        # Pasar la ruta absoluta del modelo
        manager = MasterCalibrationManager(model_path=model_path)
        
        update_progress(20)
        
        # Construir rutas
        photos_path = os.path.join(base_dir, "server", "operator-photo", operator_id)
        
        # Verificar que existe el directorio
        if not os.path.exists(photos_path):
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No existe el directorio {photos_path}\n")
            print(f"ERROR: No existe el directorio de fotos para el operador {operator_id}")
            update_progress(100)
            # Restaurar directorio antes de salir
            os.chdir(original_dir)
            sys.exit(1)
        
        update_progress(30)
        
        # Leer información del operador
        info_file = os.path.join(photos_path, "info.txt")
        operator_name = "Desconocido"
        
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Operador:"):
                            operator_name = line.replace("Operador:", "").strip()
                            break
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error leyendo info.txt: {e}\n")
        
        update_progress(40)
        
        # Log información del operador
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Operador: {operator_name} (ID: {operator_id})\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directorio de fotos: {photos_path}\n")
        
        # Ejecutar calibración
        update_progress(50)
        
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ejecutando calibración...\n")
        
        success = manager.calibrate_operator(operator_id, photos_path, operator_name)
        
        update_progress(90)
        
        if success:
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calibración completada exitosamente\n")
            print("SUCCESS")
        else:
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Calibración fallida\n")
            print("ERROR")
        
        update_progress(100)
        
    except Exception as e:
        # Log del error
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {str(e)}\n")
        
        print(f"ERROR: {str(e)}")
        update_progress(100)

if __name__ == "__main__":
    main()