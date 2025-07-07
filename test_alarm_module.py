import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alarm_module import AlarmModule

# Crear y inicializar
alarm = AlarmModule("assets/audio")
alarm.initialize()

print("Probando reproducir telefono.mp3...")

# Ver qué métodos tiene el AlarmModule
print("\nMétodos disponibles en AlarmModule:")
methods = [method for method in dir(alarm) if not method.startswith('_')]
for method in methods:
    print(f"  - {method}")

# Intentar reproducir
print("\nIntentando reproducir...")
result = alarm.play_alarm_threaded("telefono.mp3")
print(f"Resultado: {result}")

# Esperar para escuchar
print("Esperando 5 segundos...")
time.sleep(5)

print("\nFin del test")