"""
Debug para encontrar por qué el audio suena a los 4 segundos
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distraction import DistractionDetector

# Crear detector
detector = DistractionDetector()

print("\n=== SIMULACIÓN DE TIMING ===")
print(f"FPS configurado: {detector.config['camera_fps']}")
print(f"Level 1 debería sonar en frame: {detector.level1_threshold}")
print(f"Level 2 debería sonar en frame: {detector.level2_threshold}")

# Simular el contador
print("\nSimulando frames...")
detector.direction = "EXTREMO"  # Forzar estado de distracción

for frame in range(1, 35):
    # Incrementar contador manualmente
    detector.distraction_counter = frame
    tiempo_actual = detector.distraction_counter / detector.config['camera_fps']
    
    # Verificar condiciones EXACTAS del código
    if detector.distraction_counter == detector.level1_threshold and detector.current_alert_level == 0:
        print(f"Frame {frame} ({tiempo_actual:.1f}s) → NIVEL 1 ACTIVADO")
        detector.current_alert_level = 1
        
    elif detector.distraction_counter == detector.level2_threshold and detector.current_alert_level == 1:
        print(f"Frame {frame} ({tiempo_actual:.1f}s) → NIVEL 2 ACTIVADO")
        detector.current_alert_level = 2
        
    # También verificar si hay alguna otra condición
    if frame == 16:  # 4 segundos a 4 FPS
        print(f"Frame {frame} ({tiempo_actual:.1f}s) → ¿Algo especial aquí?")
    
    # Mostrar cada segundo
    if frame % 4 == 0:
        print(f"Frame {frame} = {tiempo_actual:.1f} segundos")

print("\n=== VERIFICACIÓN DE INTEGRATED SYSTEM ===")
from core.distraction import IntegratedDistractionSystem

system = IntegratedDistractionSystem()
system.set_operator({'id': 'test', 'name': 'Test'})

print(f"\nDespués de IntegratedSystem:")
print(f"detector.level1_threshold: {system.detector.level1_threshold}")
print(f"detector.level2_threshold: {system.detector.level2_threshold}")
print(f"detector.config['level2_time']: {system.detector.config['level2_time']}")