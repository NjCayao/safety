# expresiones_faciales/ir_processor.py
import cv2
import numpy as np

class IRProcessor:
    """Procesador de imágenes para cámara infrarroja"""
    
    def __init__(self, config=None):
        # Valores por defecto si no se proporciona configuración
        self.enabled = True
        self.ir_brightness_threshold = 40
        self.ir_contrast_boost = 1.5
        self.auto_switch_mode = True
        self.noise_reduction = 2
        
        # Actualizar con configuración si se proporciona
        if config:
            if 'enabled' in config:
                self.enabled = config['enabled']
            if 'ir_brightness_threshold' in config:
                self.ir_brightness_threshold = config['ir_brightness_threshold']
            if 'ir_contrast_boost' in config:
                self.ir_contrast_boost = config['ir_contrast_boost']
            if 'auto_switch_mode' in config:
                self.auto_switch_mode = config['auto_switch_mode']
            if 'noise_reduction' in config:
                self.noise_reduction = config['noise_reduction']
        
        # Estado interno
        self.is_ir_mode = False
        self.light_level = 0
        
    def process(self, frame):
        """Procesa un frame para optimizarlo según condiciones de luz"""
        if not self.enabled:
            return frame
            
        # Detectar nivel de luz
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        self.light_level = np.mean(gray)
        
        # Determinar si estamos en modo IR
        previous_mode = self.is_ir_mode
        if self.auto_switch_mode:
            self.is_ir_mode = self.light_level < self.ir_brightness_threshold
            
        # Procesamiento según modo
        if self.is_ir_mode:
            return self._process_ir_mode(frame)
        else:
            return frame  # No se procesa en modo normal
    
    def _process_ir_mode(self, frame):
        """Procesa frame en modo infrarrojo"""
        # Convertir a escala de grises si no lo está
        if len(frame.shape) == 3:
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed = frame.copy()
            
        # Mejorar contraste
        if self.ir_contrast_boost > 1.0:
            processed = self._enhance_contrast(processed)
            
        # Reducir ruido
        if self.noise_reduction > 0:
            processed = self._reduce_noise(processed)
            
        return processed
    
    def _enhance_contrast(self, image):
        """Mejora el contraste de la imagen"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        
    def _reduce_noise(self, image):
        """Reduce ruido en la imagen"""
        return cv2.GaussianBlur(image, (5, 5), self.noise_reduction)