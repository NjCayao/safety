# Guía de Integración de Sincronización Online/Offline

Esta guía explica cómo integrar la funcionalidad de sincronización online/offline con los módulos de detección existentes.

## Estructura de archivos

Los siguientes archivos se han añadido al proyecto:

- `sync_integrator.py` - Componente principal para la integración
- `fatigue_adapter.py` - Adaptador para el detector de fatiga
- `behavior_adapter.py` - Adaptador para el detector de comportamientos
- `face_recognition_adapter.py` - Adaptador para el reconocedor facial
- `main_with_sync.py` - Ejemplo de sistema principal con sincronización

## Uso básico

Para usar la sincronización con tu sistema existente:

1. Inicia el integrador de sincronización antes que el sistema principal:

```python
from sync_integrator import SyncIntegrator

# Inicializar integrador
sync_integrator = SyncIntegrator()
sync_integrator.start()