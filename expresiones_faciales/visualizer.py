import cv2
import numpy as np
from expresiones_faciales.config import VISUALIZATION_CONFIG

class Visualizer:
    """Visualizador de métricas en pantalla"""
    
    def __init__(self):
        # Cargar configuración
        self.show_landmarks = VISUALIZATION_CONFIG["show_landmarks"]
        self.show_expression = VISUALIZATION_CONFIG["show_expression"]
        self.show_fatigue = VISUALIZATION_CONFIG["show_fatigue"]
        self.show_stress = VISUALIZATION_CONFIG["show_stress"]
        self.text_size = VISUALIZATION_CONFIG["text_size"]
        self.text_color = VISUALIZATION_CONFIG["text_color"]
        self.background_opacity = VISUALIZATION_CONFIG["background_opacity"]
        self.position = VISUALIZATION_CONFIG["position"]
        
    def update_config(self, new_config):
        """Actualiza la configuración en tiempo real"""
        for key, value in new_config.items():
            if key in VISUALIZATION_CONFIG:
                VISUALIZATION_CONFIG[key] = value
                
        # Actualizar variables locales
        self.show_landmarks = VISUALIZATION_CONFIG["show_landmarks"]
        self.show_expression = VISUALIZATION_CONFIG["show_expression"]
        self.show_fatigue = VISUALIZATION_CONFIG["show_fatigue"]
        self.show_stress = VISUALIZATION_CONFIG["show_stress"]
        self.text_size = VISUALIZATION_CONFIG["text_size"]
        self.text_color = VISUALIZATION_CONFIG["text_color"]
        self.background_opacity = VISUALIZATION_CONFIG["background_opacity"]
        self.position = VISUALIZATION_CONFIG["position"]
        
    def draw_metrics(self, frame, operator_info):
        """Dibuja métricas en el frame"""
        if not operator_info:
            return frame
            
        # Crear copia para no modificar el original
        display_frame = frame.copy()
        
        # Dibujar landmarks si está habilitado
        if self.show_landmarks and 'face_landmarks' in operator_info:
            display_frame = self._draw_landmarks(display_frame, operator_info['face_landmarks'])
            
        # Determinar posición para información textual
        text_position = self._get_text_position(display_frame, operator_info)
        
        # Preparar textos a mostrar
        texts = []
        
        # Nombre del operador (siempre se muestra)
        if 'name' in operator_info:
            texts.append(f"Operador: {operator_info['name']}")
            
        # Expresión
        if self.show_expression and 'expression' in operator_info:
            texts.append(f"Expresión: {operator_info['expression']}")
            
        # Fatiga
        if self.show_fatigue and 'fatigue_percentage' in operator_info:
            fatigue = operator_info['fatigue_percentage']
            texts.append(f"Fatiga: {fatigue}%")
            
        # Estrés
        if self.show_stress and 'stress_percentage' in operator_info:
            stress = operator_info['stress_percentage']
            texts.append(f"Estrés: {stress}%")
            
        # Dibujar panel de información
        if texts:
            display_frame = self._draw_info_panel(display_frame, text_position, texts)
            
        return display_frame
    
    def _draw_landmarks(self, frame, landmarks):
        """Dibuja landmarks faciales"""
        # Implementación de dibujo
        # ...
        
    def _get_text_position(self, frame, operator_info):
        """Determina posición para texto según configuración"""
        # Implementación para determinar posición
        # ...
        
    def _draw_info_panel(self, frame, position, texts):
        """Dibuja panel con información textual"""
        # Implementación de dibujo de panel
        # ...