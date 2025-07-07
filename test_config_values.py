# test_config_values.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.behavior.behavior_detection_module import BehaviorDetectionModule

# Crear instancia
detector = BehaviorDetectionModule()

# Ver qué valores tiene audio_keys
print("Valores en audio_keys:")
for key, value in detector.audio_keys.items():
    print(f"  {key}: '{value}'")

# Probar reproducción directa
from core.alarm_module import AlarmModule
alarm = AlarmModule("assets/audio")
alarm.initialize()

print("\nProbando reproducción directa:")
print("1. Con 'telefono':")
alarm.play_audio("telefono")

import time
time.sleep(2)

print("2. Con 'telefono.mp3':")
alarm.play_audio("telefono.mp3")