# expresiones_faciales/config.py

# Configuración para análisis de expresiones
EXPRESSION_CONFIG = {
    "enabled": True,                      # Activar/desactivar análisis de expresiones
    "sensitivity": 0.7,                   # Sensibilidad para detección (0.0-1.0)
    "memory_frames": 10,                  # Frames para memoria de expresiones
    "expression_threshold": 0.65,         # Umbral para clasificación
    "show_expression": True,              # Mostrar expresión en pantalla
    "expression_position": "top-right"    # Posición en pantalla
}

# Configuración para detección de fatiga
FATIGA_CONFIG = {
    "enabled": True,                      # Activar/desactivar detección
    "sensitivity": 0.8,                   # Sensibilidad (0.0-1.0)
    "blink_threshold": 0.3,               # Umbral para detección de parpadeo
    "eye_aspect_ratio_threshold": 0.25,   # Umbral para ojos semicerrados
    "time_window_seconds": 60,            # Ventana de tiempo para análisis
    "show_percentage": True,              # Mostrar porcentaje en pantalla
    "alert_threshold": 70                 # Umbral para alertas (%)
}

# Configuración para análisis de estrés
ESTRES_CONFIG = {
    "enabled": True,                      # Activar/desactivar detección
    "sensitivity": 0.75,                  # Sensibilidad (0.0-1.0)
    "facial_tension_threshold": 0.6,      # Umbral para tensión facial
    "micro_movement_threshold": 0.3,      # Umbral para micro-movimientos
    "time_window_seconds": 30,            # Ventana para análisis
    "show_percentage": True,              # Mostrar porcentaje en pantalla
    "alert_threshold": 75                 # Umbral para alertas (%)
}

# Configuración para cámara infrarroja
IR_CONFIG = {
    "enabled": True,                      # Activar/desactivar modo IR
    "ir_brightness_threshold": 40,        # Umbral para modo automático
    "ir_contrast_boost": 1.5,             # Factor de mejora de contraste
    "auto_switch_mode": True,             # Cambio automático normal/IR
    "noise_reduction": 2                  # Nivel de reducción de ruido (0-5)
}

# Visualización
VISUALIZATION_CONFIG = {
    "show_landmarks": True,               # Mostrar puntos de referencia facial
    "show_expression": True,              # Mostrar expresión detectada
    "show_fatigue": True,                 # Mostrar nivel de fatiga
    "show_stress": True,                  # Mostrar nivel de estrés
    "text_size": 0.6,                     # Tamaño de texto (0.0-1.0)
    "text_color": (255, 255, 255),        # Color de texto (BGR)
    "background_opacity": 0.7,            # Opacidad de fondo (0.0-1.0)
    "position": "right"                   # Posición (right, left, top, bottom)
}