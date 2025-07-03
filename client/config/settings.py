# Configuración general del sistema
SYSTEM_NAME = "Safety System"
VERSION = "1.0.0"

# Configuración de la cámara
CAMERA_INDEX = 0  # Índice de la cámara (0 para la primera cámara)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_RATE = 15

# Parámetros de detección de fatiga
EYE_AR_THRESH = 0.25  # Umbral para detección de ojos cerrados
EYE_AR_CONSEC_FRAMES = 30  # Equivalente a ~2 segundos con 15 FPS

# Parámetros de reconocimiento facial
FACE_RECOGNITION_TOLERANCE = 0.6
FACE_RECOGNITION_MODEL = "hog"  # alternativa: "cnn" (requiere GPU)

# Parámetros de alertas
ALERT_INTERVAL = 10800  # 3 horas en segundos
PAUSE_DURATION = 600    # 10 minutos en segundos

# Configuración del servidor
SERVER_API_URL = "http://your-server-address/api"
API_KEY = "your-api-key"

# Directorios
OPERATORS_DIR = "operators/"
LOGS_DIR = "logs/"
REPORTS_DIR = "reports/"
AUDIO_DIR = "audio/"
MODELS_DIR = "models/"

# Nombre de archivos de audio para alertas
AUDIO_FATIGUE = "fatigue_alert.mp3"
AUDIO_PHONE = "phone_alert.mp3"
AUDIO_SMOKING = "smoking_alert.mp3"
AUDIO_BREAK = "break_alert.mp3"
AUDIO_UNAUTHORIZED = "unauthorized_alert.mp3"