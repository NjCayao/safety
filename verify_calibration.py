"""
verify_calibration.py
====================
Script para verificar el estado de calibración de operadores
"""

import os
import json
import sys

def verify_operator_calibration(operator_id):
    """Verifica y muestra el estado de calibración de un operador"""
    print(f"\n=== VERIFICACIÓN DE CALIBRACIÓN: {operator_id} ===\n")
    
    # Rutas
    photos_path = f"server/operator-photo/{operator_id}"
    baseline_path = f"operators/baseline-json/{operator_id}"
    
    # 1. Verificar fotos
    print("1. FOTOS DEL OPERADOR:")
    if os.path.exists(photos_path):
        photos = [f for f in os.listdir(photos_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   ✓ Carpeta encontrada: {photos_path}")
        print(f"   ✓ Fotos encontradas: {len(photos)}")
        
        # Leer info.txt
        info_file = os.path.join(photos_path, "info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                operator_name = f.readline().strip().replace("Operador:", "").strip()
                print(f"   ✓ Nombre: {operator_name}")
        else:
            print("   ⚠ No se encontró info.txt")
    else:
        print(f"   ✗ No existe carpeta de fotos: {photos_path}")
        return
    
    # 2. Verificar calibración maestra
    print("\n2. CALIBRACIÓN MAESTRA:")
    master_file = os.path.join(baseline_path, "master_baseline.json")
    if os.path.exists(master_file):
        print(f"   ✓ Archivo encontrado: {master_file}")
        with open(master_file, 'r') as f:
            master_data = json.load(f)
        
        print(f"   - Fotos procesadas: {master_data['calibration_info']['photos_processed']}")
        print(f"   - Fecha creación: {master_data['calibration_info']['created_at']}")
        
        if 'statistics' in master_data and 'avg_ear' in master_data['statistics']:
            ear_stats = master_data['statistics']['avg_ear']
            print(f"   - EAR promedio: {ear_stats['mean']:.3f}")
            print(f"   - EAR rango: [{ear_stats['min']:.3f} - {ear_stats['max']:.3f}]")
    else:
        print(f"   ✗ No existe calibración maestra")
    
    # 3. Verificar calibración de fatiga
    print("\n3. CALIBRACIÓN DE FATIGA:")
    fatigue_file = os.path.join(baseline_path, "fatigue_baseline.json")
    if os.path.exists(fatigue_file):
        print(f"   ✓ Archivo encontrado: {fatigue_file}")
        with open(fatigue_file, 'r') as f:
            fatigue_data = json.load(f)
        
        thresholds = fatigue_data['thresholds']
        print(f"   - Umbral EAR: {thresholds['ear_threshold']:.3f}")
        print(f"   - Ajuste nocturno: {thresholds['ear_night_adjustment']:.3f}")
        print(f"   - Umbral microsueño: {thresholds['microsleep_threshold']}s")
        print(f"   - Confianza: {thresholds['calibration_confidence']*100:.1f}%")
        
        # Mostrar estadísticas si existen
        if 'statistics' in fatigue_data and 'ear_stats' in fatigue_data['statistics']:
            stats = fatigue_data['statistics']['ear_stats']
            print(f"\n   ESTADÍSTICAS EAR:")
            print(f"   - Media: {stats['mean']:.3f}")
            print(f"   - Desv. estándar: {stats['std']:.3f}")
            print(f"   - Percentiles: P25={stats['percentiles']['25']:.3f}, "
                  f"P50={stats['percentiles']['50']:.3f}, "
                  f"P75={stats['percentiles']['75']:.3f}")
    else:
        print(f"   ✗ No existe calibración de fatiga")
    
    # 4. Verificar otros módulos
    print("\n4. OTROS MÓDULOS:")
    other_files = {
        "behavior_baseline.json": "Comportamientos",
        "distraction_baseline.json": "Distracciones",
        "yawn_baseline.json": "Bostezos"
    }
    
    for filename, module in other_files.items():
        file_path = os.path.join(baseline_path, filename)
        if os.path.exists(file_path):
            print(f"   ✓ {module}: Calibrado")
        else:
            print(f"   ⚠ {module}: No calibrado")
    
    print("\n" + "="*50)


def list_all_operators():
    """Lista todos los operadores disponibles"""
    print("\n=== OPERADORES DISPONIBLES ===\n")
    
    # Buscar en fotos
    photos_dir = "server/operator-photo"
    if os.path.exists(photos_dir):
        operators_photos = set(os.listdir(photos_dir))
        print(f"Operadores con fotos: {len(operators_photos)}")
    else:
        operators_photos = set()
        print("No se encontró directorio de fotos")
    
    # Buscar en calibraciones
    baseline_dir = "operators/baseline-json"
    if os.path.exists(baseline_dir):
        operators_calibrated = set(os.listdir(baseline_dir))
        print(f"Operadores calibrados: {len(operators_calibrated)}")
    else:
        operators_calibrated = set()
        print("No se encontró directorio de calibraciones")
    
    # Mostrar tabla
    all_operators = operators_photos.union(operators_calibrated)
    
    if all_operators:
        print("\n{:<15} {:<20} {:<15}".format("ID", "Fotos", "Calibrado"))
        print("-" * 50)
        
        for op_id in sorted(all_operators):
            has_photos = "✓" if op_id in operators_photos else "✗"
            is_calibrated = "✓" if op_id in operators_calibrated else "✗"
            print(f"{op_id:<15} {has_photos:<20} {is_calibrated:<15}")
    else:
        print("\nNo se encontraron operadores")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        operator_id = sys.argv[1]
        verify_operator_calibration(operator_id)
    else:
        # Listar todos los operadores
        list_all_operators()
        
        print("\nUso: python verify_calibration.py [ID_OPERADOR]")
        print("Ejemplo: python verify_calibration.py 47469940")