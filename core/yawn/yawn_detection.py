"""
Módulo de Detección de Bostezos
===============================
Detecta bostezos usando MAR (Mouth Aspect Ratio) con calibración personalizada.
"""

import cv2
import numpy as np
import time
import logging
import pygame
from scipy.spatial import distance
from collections import deque

# Importar configuración si está disponible
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

class YawnDetector:
    def __init__(self, config=None):
        """
        Inicializa el detector de bostezos.
        
        Args:
            config: Configuración personalizada (opcional)
        """
        self.logger = logging.getLogger('YawnDetector')
        
        # Configuración base
        if config:
            self.config = config
        else:
            # Cargar configuración por defecto
            self.config = {
                'mar_threshold': 0.7,
                'duration_threshold': 2.5,
                'frames_to_confirm': 3,
                'night_mode_threshold': 50,
                'night_adjustment': 0.05,
                'enable_night_mode': True,
                'enable_sounds': True,
                'calibration_confidence': 0.0
            }
        
        # Variables de estado
        self.yawn_in_progress = False
        self.yawn_start_time = None
        self.yawn_counter = 0
        self.normal_counter = 0
        self.last_mar_values = deque(maxlen=3)
        
        # Estado de iluminación
        self.is_night_mode = False
        self.light_level = 0
        
        # Colores para visualización
        self.colors = {
            'mouth_normal': (0, 255, 0),
            'mouth_yawning': (0, 0, 255),
            'text': (255, 255, 255),
            'background': (0, 0, 0)
        }
        
        # Inicializar pygame para audio
        if self.config['enable_sounds']:
            try:
                pygame.mixer.init()
                self.audio_initialized = True
            except:
                self.logger.warning("No se pudo inicializar el sistema de audio")
                self.audio_initialized = False
        else:
            self.audio_initialized = False
        
        self.logger.info("YawnDetector inicializado")
    
    def update_config(self, new_config):
        """
        Actualiza la configuración del módulo.
        
        Args:
            new_config: Diccionario con nueva configuración
        """
        self.config.update(new_config)
        self.logger.info("Configuración actualizada")
    
    def detect(self, frame, landmarks):
        """
        Detecta bostezos en el frame actual.
        
        Args:
            frame: Frame de video
            landmarks: Landmarks faciales detectados
            
        Returns:
            dict: Información de la detección
        """
        if landmarks is None:
            return self._create_empty_result()
        
        # Detectar condiciones de iluminación
        if self.config['enable_night_mode']:
            self._detect_lighting_conditions(frame)
        
        # Ajustar umbral según modo día/noche
        current_threshold = self.config['mar_threshold']
        if self.is_night_mode:
            current_threshold -= self.config['night_adjustment']
        
        # Extraer puntos de la boca
        mouth_points = self._get_mouth_points(landmarks)
        
        # Calcular MAR
        mar = self._calculate_mar(mouth_points)
        self.last_mar_values.append(mar)
        avg_mar = sum(self.last_mar_values) / len(self.last_mar_values)
        
        # Detectar bostezo
        current_yawn = avg_mar > current_threshold
        
        # Suavizar detección
        if current_yawn:
            self.yawn_counter += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            self.yawn_counter = 0
        
        # Confirmar detección
        confirmed_yawn = self.yawn_counter >= self.config['frames_to_confirm']
        confirmed_normal = self.normal_counter >= self.config['frames_to_confirm']
        
        # Procesar estado del bostezo
        current_time = time.time()
        yawn_detected = False
        yawn_duration = 0
        
        if confirmed_yawn and not self.yawn_in_progress:
            # Inicio de bostezo
            self.yawn_in_progress = True
            self.yawn_start_time = current_time
            self.logger.info(f"Inicio de bostezo detectado (MAR: {mar:.2f})")
            
        elif confirmed_normal and self.yawn_in_progress:
            # Fin de bostezo
            self.yawn_in_progress = False
            yawn_duration = current_time - self.yawn_start_time
            
            if yawn_duration >= self.config['duration_threshold']:
                yawn_detected = True
                self.logger.info(f"Bostezo completado: {yawn_duration:.1f}s")
        
        # Si está en progreso, calcular duración actual
        if self.yawn_in_progress and self.yawn_start_time:
            yawn_duration = current_time - self.yawn_start_time
        
        # Crear resultado
        result = {
            'mar_value': mar,
            'smooth_mar': avg_mar,
            'mar_threshold': current_threshold,
            'is_yawning': self.yawn_in_progress,
            'yawn_detected': yawn_detected,
            'yawn_duration': yawn_duration,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'mouth_points': mouth_points,
            'timestamp': current_time
        }
        
        return result
    
    def draw_yawn_info(self, frame, result):
        """
        Dibuja información del bostezo en el frame.
        ACTUALIZADO: Contorno verde + puntos rojos SIN borde blanco
        
        Args:
            frame: Frame donde dibujar
            result: Resultado de la detección
            
        Returns:
            frame: Frame con información dibujada
        """
        if not result or 'mouth_points' not in result:
            return frame
        
        # Dibujar contorno de la boca
        mouth_points = result['mouth_points']
        if mouth_points and len(mouth_points) > 0:
            # CONTORNO VERDE
            hull = cv2.convexHull(np.array(mouth_points))
            color_contorno = self.colors['mouth_normal']  # Verde siempre
            cv2.drawContours(frame, [hull], -1, color_contorno, 2)
            
            # PUNTOS ROJOS (sin borde blanco)
            color_puntos = (0, 0, 255)  # Rojo para todos los puntos
            for (x, y) in mouth_points:
                # Solo punto rojo, sin borde
                cv2.circle(frame, (x, y), 3, color_puntos, -1)
        
        # Información de texto
        h, w = frame.shape[:2]
        y_offset = h - 120
        
        # Estado actual
        status = "BOSTEZANDO" if result['is_yawning'] else "Normal"
        status_color = self.colors['mouth_yawning'] if result['is_yawning'] else self.colors['mouth_normal']
        cv2.putText(frame, f"Estado: {status}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Valor MAR
        y_offset += 30
        mode_str = "NOCHE" if result['is_night_mode'] else "DÍA"
        cv2.putText(frame, f"MAR: {result['mar_value']:.2f} (Umbral: {result['mar_threshold']:.2f}, Modo: {mode_str})",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Duración si está bostezando
        if result['is_yawning'] and result['yawn_duration'] > 0:
            y_offset += 30
            progress = min(result['yawn_duration'] / self.config['duration_threshold'], 1.0) * 100
            cv2.putText(frame, f"Duración: {result['yawn_duration']:.1f}s ({progress:.0f}%)",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['mouth_yawning'], 2)
            
            # Barra de progreso
            bar_width = 200
            bar_height = 15
            y_offset += 5
            cv2.rectangle(frame, (10, y_offset), (10 + bar_width, y_offset + bar_height),
                        (100, 100, 100), -1)
            filled_width = int(bar_width * progress / 100)
            cv2.rectangle(frame, (10, y_offset), (10 + filled_width, y_offset + bar_height),
                        self.colors['mouth_yawning'], -1)
        
        return frame
            
    def _detect_lighting_conditions(self, frame):
        """Detecta las condiciones de iluminación"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.light_level = np.mean(gray)
        
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.config['night_mode_threshold']
        
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
            self.logger.info(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
    
    def _get_mouth_points(self, landmarks):
        """Extrae los puntos de la boca desde los landmarks"""
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
    
    def _calculate_mar(self, mouth_points):
        """Calcula Mouth Aspect Ratio con detección mejorada anti-sonrisas"""
        if len(mouth_points) < 20:
            return 0
        
        # Calcular varias alturas
        heights = []
        
        # Altura en varios puntos
        heights.append(abs(mouth_points[14][1] - mouth_points[18][1]))
        heights.append(abs(mouth_points[3][1] - mouth_points[9][1]))
        heights.append(abs(mouth_points[15][1] - mouth_points[19][1]))
        heights.append(abs(mouth_points[4][1] - mouth_points[8][1]))
        heights.append(abs(mouth_points[16][1] - mouth_points[17][1]))
        
        # Usar el máximo de las alturas
        mouth_height = max(heights)
        
        # Calcular ancho
        mouth_width = distance.euclidean(mouth_points[0], mouth_points[6])
        
        # MAR básico
        mar = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # === DETECCIÓN MEJORADA DE SONRISA/RISA ===
        
        # 1. Detectar curvatura de los labios
        # En una sonrisa, las comisuras (extremos) suben respecto al centro
        left_corner = mouth_points[0]   # Comisura izquierda
        right_corner = mouth_points[6]  # Comisura derecha
        
        # Puntos centrales del labio superior e inferior
        upper_center = mouth_points[3]  # Centro labio superior
        lower_center = mouth_points[9]  # Centro labio inferior
        
        # Calcular elevación de comisuras respecto al centro
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        avg_center_y = (upper_center[1] + lower_center[1]) / 2
        
        # Si las comisuras están más arriba que el centro, es sonrisa
        corner_elevation = avg_center_y - avg_corner_y  # Positivo = comisuras elevadas
        
        # 2. Calcular asimetría (las sonrisas suelen ser más simétricas)
        left_heights = [heights[0], heights[1]]  # Alturas lado izquierdo
        right_heights = [heights[3], heights[4]]  # Alturas lado derecho
        asymmetry = abs(np.mean(left_heights) - np.mean(right_heights)) / mouth_height if mouth_height > 0 else 0
        
        # 3. Calcular forma de la apertura
        # En un bostezo, la altura central es mayor que en los extremos
        center_height = heights[2]  # Altura central
        edge_heights = (heights[0] + heights[4]) / 2  # Promedio alturas extremos
        center_dominance = (center_height - edge_heights) / center_height if center_height > 0 else 0
        
        # 4. Detectar si es sonrisa/risa
        is_smiling = False
        
        # Criterios para sonrisa:
        # - Comisuras elevadas significativamente
        # - Boca más ancha que alta
        # - Poca dominancia central
        if corner_elevation > 5:  # Comisuras elevadas más de 5 píxeles
            is_smiling = True
            self.logger.debug(f"Sonrisa detectada: comisuras elevadas {corner_elevation:.1f}px")
        
        elif mouth_width / mouth_height > 3.5 and center_dominance < 0.2:
            # Boca muy ancha y plana
            is_smiling = True
            self.logger.debug(f"Sonrisa detectada: ratio ancho/alto = {mouth_width/mouth_height:.1f}")
        
        elif mar > 0.3 and mar < 0.45 and asymmetry > 0.3:
            # Apertura moderada pero asimétrica (sonrisa ladeada)
            is_smiling = True
            self.logger.debug(f"Sonrisa detectada: asimetría = {asymmetry:.2f}")
        
        # 5. Ajustar MAR si es sonrisa
        original_mar = mar
        if is_smiling:
            # Penalizar fuertemente las sonrisas
            mar = mar * 0.6  # Reducir 40%
            self.logger.info(f"Sonrisa/risa filtrada: MAR {original_mar:.3f} -> {mar:.3f}")
        
        # 6. Boost para bostezos reales
        # Los bostezos tienen apertura central dominante
        elif center_dominance > 0.15 and mar > 0.35:
            # Es probable un bostezo real
            mar = mar * 1.1  # Aumentar 10%
            self.logger.debug(f"Bostezo reforzado: dominancia central = {center_dominance:.2f}")
        
        return mar
    
    def _create_empty_result(self):
        """Crea un resultado vacío cuando no hay detección"""
        return {
            'mar_value': 0,
            'smooth_mar': 0,
            'mar_threshold': self.config['mar_threshold'],
            'is_yawning': False,
            'yawn_detected': False,
            'yawn_duration': 0,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'mouth_points': [],
            'timestamp': time.time()
        }
    
    def get_status(self):
        """Obtiene el estado actual del detector"""
        return {
            'is_yawning': self.yawn_in_progress,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'calibration_confidence': self.config.get('calibration_confidence', 0),
            'current_threshold': self.config['mar_threshold'] - (
                self.config['night_adjustment'] if self.is_night_mode else 0
            )
        }
    
    def reset(self):
        """Reinicia el detector"""
        self.yawn_in_progress = False
        self.yawn_start_time = None
        self.yawn_counter = 0
        self.normal_counter = 0
        self.last_mar_values.clear()
        self.logger.info("Detector de bostezos reiniciado")