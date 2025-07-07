"""
adjust_calibration.py
====================
Script para ajustar manualmente la calibración de fatiga
"""

import json
import os
import sys

def adjust_fatigue_calibration(operator_id, adjustments):
    """
    Ajusta la calibración de fatiga para un operador
    
    Args:
        operator_id: ID del operador
        adjustments: Dict con ajustes a aplicar
    """
    baseline_path = f"operators/baseline-json/{operator_id}/fatigue_baseline.json"
    
    if not os.path.exists(baseline_path):
        print(f"✗ No existe calibración para operador {operator_id}")
        return False
    
    # Cargar calibración actual
    with open(baseline_path, 'r') as f:
        calibration = json.load(f)
    
    print(f"\n=== CALIBRACIÓN ACTUAL para {operator_id} ===")
    print(f"Umbral EAR: {calibration['thresholds']['ear_threshold']:.3f}")
    print(f"Tiempo microsueño: {calibration['thresholds']['microsleep_threshold']}s")
    print(f"Frames confirmación: {calibration['thresholds'].get('frames_to_confirm', 2)}")
    
    # Aplicar ajustes
    for key, value in adjustments.items():
        if key in calibration['thresholds']:
            old_value = calibration['thresholds'][key]
            calibration['thresholds'][key] = value
            print(f"\n✓ {key}: {old_value} → {value}")
    
    # Agregar nota de ajuste manual
    calibration['calibration_info']['manual_adjustment'] = {
        'timestamp': datetime.now().isoformat(),
        'adjustments': adjustments
    }
    
    # Guardar calibración actualizada
    with open(baseline_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\n✓ Calibración actualizada guardada")
    return True


def interactive_adjustment(operator_id):
    """Ajuste interactivo de calibración"""
    print(f"\n=== AJUSTE INTERACTIVO DE CALIBRACIÓN ===")
    print(f"Operador: {operator_id}")
    
    # Cargar calibración actual
    baseline_path = f"operators/baseline-json/{operator_id}/fatigue_baseline.json"
    if not os.path.exists(baseline_path):
        print(f"✗ No existe calibración para este operador")
        return
    
    with open(baseline_path, 'r') as f:
        calibration = json.load(f)
    
    current_ear = calibration['thresholds']['ear_threshold']
    current_time = calibration['thresholds']['microsleep_threshold']
    
    print(f"\nValores actuales:")
    print(f"1. Umbral EAR: {current_ear:.3f}")
    print(f"2. Tiempo microsueño: {current_time}s")
    print(f"3. Frames confirmación: {calibration['thresholds'].get('frames_to_confirm', 2)}")
    
    print("\n¿Qué problema tienes?")
    print("A. Detecta microsueños con parpadeos normales (muy sensible)")
    print("B. No detecta microsueños reales (poco sensible)")
    print("C. Ajuste manual personalizado")
    print("Q. Salir sin cambios")
    
    choice = input("\nElige una opción: ").upper()
    
    adjustments = {}
    
    if choice == 'A':
        # Hacer menos sensible
        print("\nHaciendo calibración MENOS sensible...")
        adjustments['ear_threshold'] = max(0.20, current_ear * 0.85)  # Reducir umbral 15%
        adjustments['microsleep_threshold'] = 2.0  # Aumentar tiempo requerido
        adjustments['frames_to_confirm'] = 4  # Más frames para confirmar
        
    elif choice == 'B':
        # Hacer más sensible
        print("\nHaciendo calibración MÁS sensible...")
        adjustments['ear_threshold'] = min(0.35, current_ear * 1.15)  # Aumentar umbral 15%
        adjustments['microsleep_threshold'] = 1.2  # Reducir tiempo requerido
        adjustments['frames_to_confirm'] = 2  # Menos frames para confirmar
        
    elif choice == 'C':
        # Ajuste manual
        print("\nAJUSTE MANUAL")
        try:
            new_ear = float(input(f"Nuevo umbral EAR (actual: {current_ear:.3f}): "))
            adjustments['ear_threshold'] = max(0.15, min(0.40, new_ear))
            
            new_time = float(input(f"Nuevo tiempo microsueño en segundos (actual: {current_time}): "))
            adjustments['microsleep_threshold'] = max(0.5, min(5.0, new_time))
            
            new_frames = int(input(f"Frames para confirmar (actual: {calibration['thresholds'].get('frames_to_confirm', 2)}): "))
            adjustments['frames_to_confirm'] = max(1, min(10, new_frames))
            
        except ValueError:
            print("✗ Valores inválidos")
            return
    
    elif choice == 'Q':
        print("Saliendo sin cambios...")
        return
    
    else:
        print("✗ Opción inválida")
        return
    
    # Aplicar ajustes
    if adjustments:
        if adjust_fatigue_calibration(operator_id, adjustments):
            print("\n✓ Calibración ajustada exitosamente")
            print("\nNuevos valores:")
            for key, value in adjustments.items():
                print(f"  - {key}: {value}")
            print("\nPrueba nuevamente el sistema para verificar los cambios")


if __name__ == "__main__":
    from datetime import datetime
    
    if len(sys.argv) > 1:
        operator_id = sys.argv[1]
        
        if len(sys.argv) > 2 and sys.argv[2] == "--auto-less-sensitive":
            # Ajuste automático para menos sensible
            adjustments = {
                'ear_threshold': 0.28,  # Más bajo = menos sensible
                'microsleep_threshold': 2.0,
                'frames_to_confirm': 3
            }
            adjust_fatigue_calibration(operator_id, adjustments)
        else:
            # Modo interactivo
            interactive_adjustment(operator_id)
    else:
        print("Uso: python adjust_calibration.py [ID_OPERADOR]")
        print("Ejemplo: python adjust_calibration.py 47469940")
        print("\nOpciones:")
        print("  --auto-less-sensitive  : Ajuste automático para reducir sensibilidad")