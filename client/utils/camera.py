import cv2
import logging

logger = logging.getLogger('Camera')

class Camera:
    def __init__(self, camera_index=0, width=640, height=480, fps=15):
        """Inicializa la cámara con los parámetros especificados"""
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.connect()
    
    def connect(self):
        """Conecta a la cámara"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir la cámara con índice {self.camera_index}")
                return False
            
            logger.info(f"Cámara inicializada correctamente: {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            logger.error(f"Error al conectar con la cámara: {str(e)}")
            return False
    
    def get_frame(self):
        """Captura y devuelve un frame de la cámara"""
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("No se pudo capturar frame")
            return None
        
        return frame
    
    def release(self):
        """Libera los recursos de la cámara"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Cámara liberada")