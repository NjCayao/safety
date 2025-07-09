import time
import os
import pygame
import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import logging  # IMPORTANTE: Asegurar este import

# 🆕 NUEVO: Importar sistema de configuración
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Sistema de configuración no disponible para DistractionDetector, usando valores por defecto")

class DistractionDetector:
    def __init__(self):
        """Inicializa el detector de distracciones con configuración centralizada"""
        
        # ===== PRIMERO: CREAR EL LOGGER =====
        self.logger = logging.getLogger('DistractionDetector')
        
        # ===== DESPUÉS: CARGAR CONFIGURACIÓN =====
        if CONFIG_AVAILABLE:
            # Configuración desde YAML
            self.config = {
                'rotation_threshold_day': get_config('distraction.rotation_threshold_day', 2.6),
                'rotation_threshold_night': get_config('distraction.rotation_threshold_night', 2.8),
                'extreme_rotation_threshold': get_config('distraction.extreme_rotation_threshold', 2.5),
                'level1_time': get_config('distraction.level1_time', 3),
                'level2_time': get_config('distraction.level2_time', 5),
                'visibility_threshold': get_config('distraction.visibility_threshold', 15),
                'frames_without_face_limit': get_config('distraction.frames_without_face_limit', 5),
                'confidence_threshold': get_config('distraction.confidence_threshold', 0.7),
                'night_mode_threshold': get_config('distraction.night_mode_threshold', 50),
                'enable_night_mode': get_config('distraction.enable_night_mode', True),
                'prediction_buffer_size': get_config('distraction.prediction_buffer_size', 10),
                'distraction_window': get_config('distraction.distraction_window', 600),
                'min_frames_for_reset': get_config('distraction.min_frames_for_reset', 10),
                'audio_enabled': get_config('distraction.audio_enabled', True),
                'level1_volume': get_config('distraction.level1_volume', 0.8),
                'level2_volume': get_config('distraction.level2_volume', 1.0),
                'camera_fps': get_config('distraction.camera_fps', 4)
            }
            self.show_gui = has_gui()
            
            print(f"✅ DistractionDetector - Configuración cargada:")
            print(f"   - Umbral rotación día: {self.config['rotation_threshold_day']}")
            print(f"   - Umbral rotación noche: {self.config['rotation_threshold_night']}")
            print(f"   - Tiempo nivel 1: {self.config['level1_time']}s")
            print(f"   - Tiempo nivel 2: {self.config['level2_time']}s")
            print(f"   - GUI: {self.show_gui}")
            print(f"   - Audio: {self.config['audio_enabled']}")
        else:
            # Configuración por defecto
            self.config = {
                'rotation_threshold_day': 2.6,
                'rotation_threshold_night': 2.8,
                'extreme_rotation_threshold': 2.5,
                'level1_time': 3,
                'level2_time': 5,
                'visibility_threshold': 15,
                'frames_without_face_limit': 5,
                'confidence_threshold': 0.7,
                'night_mode_threshold': 50,
                'enable_night_mode': True,
                'prediction_buffer_size': 10,
                'distraction_window': 600,
                'min_frames_for_reset': 10,
                'audio_enabled': True,
                'level1_volume': 0.8,
                'level2_volume': 1.0,
                'camera_fps': 4
            }
            self.show_gui = True
            print("⚠️ DistractionDetector usando configuración por defecto")
        
        # ===== CALCULAR FRAMES =====
        self.level1_threshold = int(self.config['level1_time'] * self.config['camera_fps'])
        self.level2_threshold = int(self.config['level2_time'] * self.config['camera_fps'])
        
        # ===== INICIALIZAR BUFFERS =====
        buffer_size = self.config['prediction_buffer_size']
        self.direction_buffer = deque(["CENTRO"] * buffer_size, maxlen=buffer_size)
        self.confidence_buffer = deque([1.0] * buffer_size, maxlen=buffer_size)
        
        # ===== ESTADOS =====
        self.last_valid_direction = "CENTRO"
        self.last_valid_confidence = 1.0
        self.frames_without_face = 0
        self.distraction_times = []        
        self.distraction_counter = 0       
        self.current_alert_level = 0
        self.is_night_mode = False
        self.light_level = 0
        
        # ===== VARIABLES PARA AUDIO =====
        self.alarm_module = None
        self.audio_ready = False
        
        # ===== INICIALIZAR AUDIO =====
        self._initialize_audio()
        
        # ===== VARIABLES PARA VISUALIZACIÓN =====
        self.direction = "CENTRO"          
        self.rotation_angle = 0            
        self.detection_confidence = 1.0    
        self.last_detection_time = 0       
        self.last_metrics = {}             
        self._last_log_time = 0
        
        self.logger.info("=== Detector de Distracciones Inicializado ===")
        self.logger.info(f"Tiempo Nivel 1: {self.config['level1_time']} segundos")
        self.logger.info(f"Tiempo Nivel 2: {self.config['level2_time']} segundos")
        
    def update_config(self, new_config):
        """Actualiza la configuración desde el panel web"""
        self.config.update(new_config)
        
        # Recalcular umbrales de frames
        self.level1_threshold = int(self.config['level1_time'] * self.config['camera_fps'])
        self.level2_threshold = int(self.config['level2_time'] * self.config['camera_fps'])
        
        # Actualizar volúmenes de audio
        if hasattr(self, 'level1_sound') and self.level1_sound:
            self.level1_sound.set_volume(self.config['level1_volume'])
        if hasattr(self, 'level2_sound') and self.level2_sound:
            self.level2_sound.set_volume(self.config['level2_volume'])
            
        print("Configuración actualizada desde panel web")
        
    def _initialize_audio(self):
        """Inicializa el sistema de audio con dos niveles de alerta"""
        if not self.config['audio_enabled']:
            return
            
        try:
            # Solo registrar que usaremos AlarmModule
            self.logger.info("Sistema de audio configurado para usar AlarmModule")
            self.audio_ready = True
                    
        except Exception as e:
            self.logger.error(f"❌ ERROR al configurar audio: {e}")
            self.audio_ready = False

    def _play_sound(self, level):
        """Reproduce el sonido correspondiente al nivel de alerta"""
        if not self.config['audio_enabled']:
            return
            
        try:
            # Verificar que tenemos AlarmModule
            if not self.alarm_module:
                self.logger.warning("AlarmModule no está configurado")
                return
            
            # Detener cualquier audio previo
            self.alarm_module.stop_audio()
            
            if level == 1:
                # Nivel 1: 3 segundos
                self.logger.info(f"🔊 Reproduciendo alerta nivel 1: vadelante1")
                success = self.alarm_module.play_audio("vadelante1")
                if not success:
                    self.logger.error("Error reproduciendo vadelante1")
                    
            elif level == 2:
                # Nivel 2: 7 segundos  
                self.logger.info(f"🔊 Reproduciendo alerta nivel 2: comportamiento10s")
                success = self.alarm_module.play_audio("comportamiento10s")
                if not success:
                    self.logger.error("Error reproduciendo comportamiento10s")
                    
        except Exception as e:
            self.logger.error(f"❌ Error al reproducir sonido nivel {level}: {e}")

    def set_alarm_module(self, alarm_module):
        """Configura la referencia al AlarmModule"""
        self.alarm_module = alarm_module
        self.logger.info("AlarmModule configurado en DistractionDetector")
    
    def detect(self, landmarks, frame):
        """Detecta distracciones enfocándose SOLO en giros extremos"""
        
        # Detectar condiciones de iluminación si está habilitado
        if self.config['enable_night_mode']:
            self._detect_lighting_conditions(frame)
        
        # Primero verificar si tenemos landmarks válidos
        if landmarks is None or landmarks.num_parts == 0:
            self.frames_without_face += 1
            
            # Si perdemos la cara por muchos frames, asumir giro extremo
            if self.frames_without_face > self.config['frames_without_face_limit']:
                self.direction = "EXTREMO"
                self.detection_confidence = 0.7
            else:
                # Si no hay cara y no ha pasado suficiente tiempo, no hay distracción
                self.direction = "CENTRO"
                self.detection_confidence = 0.0
            
            return self._handle_distraction_timing(frame)
        
        # Si recuperamos la cara, resetear contador
        self.frames_without_face = 0
        
        # SOLO verificar si es un giro extremo
        is_extreme_rotation = self._check_extreme_rotation(landmarks, frame)
        
        if is_extreme_rotation:
            self.direction = "EXTREMO"
            self.detection_confidence = 0.9
        else:
            # Si no es extremo, está mirando "normal" (centro, izq o der moderado)
            self.direction = "CENTRO"
            self.detection_confidence = 1.0
        
        return self._handle_distraction_timing(frame)
    
    def _check_extreme_rotation(self, landmarks, frame):
        """Verifica si hay un giro extremo de cabeza"""
        try:
            # Obtener el contorno facial
            jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
            
            # Calcular el ancho del rostro visible
            leftmost = min(point[0] for point in jaw_points)
            rightmost = max(point[0] for point in jaw_points)
            face_width = rightmost - leftmost
            
            # Calcular la altura del rostro visible
            topmost = landmarks.part(19).y  # Ceja
            bottommost = landmarks.part(8).y  # Mentón
            face_height = bottommost - topmost
            
            # En giros extremos, el rostro se ve mucho más estrecho
            aspect_ratio = face_width / face_height if face_height > 0 else 1
            
            # Verificar visibilidad de puntos clave
            nose = landmarks.part(30)
            left_eye_outer = landmarks.part(36)
            right_eye_outer = landmarks.part(45)
            
            # Distancia entre ojos externos
            eye_distance = distance.euclidean(
                (left_eye_outer.x, left_eye_outer.y),
                (right_eye_outer.x, right_eye_outer.y)
            )
            
            # En giros extremos, esta distancia se reduce drásticamente
            normal_eye_distance = face_width * 0.6  
            eye_visibility_ratio = eye_distance / normal_eye_distance if normal_eye_distance > 0 else 1
            
            # Verificar si un lado de la cara está casi oculto
            nose_x = nose.x
            face_center_x = (leftmost + rightmost) / 2
            nose_offset = abs(nose_x - face_center_x) / face_width if face_width > 0 else 0
            
            # Criterios para giro extremo usando configuración
            is_extreme = (
                aspect_ratio < 0.5 or  
                eye_visibility_ratio < 0.5 or  
                nose_offset > 0.4 or  
                face_width < self.config['visibility_threshold']
            )
            
            if is_extreme:
                # Determinar dirección del giro extremo (CORREGIDO: invertido para coincidir con vista)
                if nose_x < face_center_x:
                    self.last_valid_direction = "IZQUIERDA"  # Cambio aquí
                else:
                    self.last_valid_direction = "DERECHA"    # Cambio aquí
            
            return is_extreme
            
        except Exception as e:
            return True
    
    def _detect_normal_rotation(self, landmarks, frame):
        """Detección que IGNORA rotaciones normales, solo marca como distracción si NO es extremo"""
        
        # Si llegamos aquí, NO es un giro extremo
        # Por lo tanto, lo consideramos como posición "normal" (centro o giro moderado)
        self.direction = "CENTRO"  # Siempre centro si no es extremo
        self.detection_confidence = 1.0
        self.last_valid_direction = "CENTRO"
        self.last_valid_confidence = 1.0
        
        # NO es una distracción
        return self._handle_distraction_timing(frame)
    
    def _handle_distraction_timing(self, frame):
        """Maneja el timing de distracciones y los dos niveles de alerta"""
        
        # SOLO contar como distracción si es EXTREMO
        is_distracted = (self.direction == "EXTREMO")
        current_time = time.time()
        
        if is_distracted:
            self.distraction_counter += 1
            
            # Calcular tiempo en segundos
            tiempo_actual = self.distraction_counter / self.config['camera_fps']
            
            # Nivel 1: EXACTAMENTE a los 3 segundos (solo una vez)
            if self.distraction_counter == self.level1_threshold and self.current_alert_level == 0:
                print(f"⚠️ NIVEL 1: Giro extremo detectado ({tiempo_actual:.1f} segundos)")
                self._play_sound(1)
                self.current_alert_level = 1
            
            # Nivel 2: EXACTAMENTE a los 7 segundos (solo una vez)
            elif self.distraction_counter == self.level2_threshold and self.current_alert_level == 1:
                print(f"🚨 NIVEL 2: Giro extremo prolongado ({tiempo_actual:.1f} segundos)")
                self._play_sound(2)
                self.current_alert_level = 2
                
                # Registrar la distracción
                self.distraction_times.append(current_time)
                
                # Imprimir registro del evento
                count = len(self.distraction_times)
                print(f"📊 Giro extremo #{count} registrado")
        else:
            # Volvió a posición normal
            if self.distraction_counter > 0:
                tiempo_distraccion = self.distraction_counter / self.config['camera_fps']
                
                # Solo imprimir si fue una distracción significativa (más de 1 segundo)
                if tiempo_distraccion >= 1.0:
                    print(f"✅ Volvió a posición normal tras {tiempo_distraccion:.1f}s")
                    
                    # Si alcanzó nivel 2, confirmar que se registró
                    if self.current_alert_level >= 2:
                        print(f"   → Evento registrado. Total: {len(self.distraction_times)}/3")
            
            # Resetear contadores
            self.distraction_counter = 0
            self.current_alert_level = 0
        
        # Limpiar distracciones antiguas (más de 10 minutos)
        self.distraction_times = [t for t in self.distraction_times 
                                if t > current_time - self.config['distraction_window']]
        
        # Verificar múltiples distracciones
        multiple_distractions = len(self.distraction_times) >= 3
        
        # Dibujar visualización solo si GUI está habilitada
        if self.show_gui:
            self._draw_enhanced_visualization(frame, is_distracted)
        else:
            # En modo headless, log periódico
            if current_time - self._last_log_time > 10:
                mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                if is_distracted:
                    print(f"📊 Estado: GIRO EXTREMO | Tiempo: {self.distraction_counter/self.config['camera_fps']:.1f}s | Nivel: {self.current_alert_level} | Total: {len(self.distraction_times)}/3 | Modo: {mode_str}")
                self._last_log_time = current_time
        
        # SIEMPRE retornar una tupla
        return is_distracted, multiple_distractions
    
    def _play_sound(self, level):
        """Reproduce el sonido correspondiente al nivel de alerta usando AlarmModule"""
        if not self.config['audio_enabled']:
            return
            
        try:
            # Verificar que tenemos AlarmModule configurado
            if not hasattr(self, 'alarm_module') or self.alarm_module is None:
                self.logger.warning("AlarmModule no está configurado - no se puede reproducir audio")
                return
            
            if level == 1:
                # Nivel 1: 3 segundos - vadelante1
                self.logger.info(f"🔊 Solicitando reproducción nivel 1: vadelante1")
                success = self.alarm_module.play_audio("vadelante1")
                if success:
                    print(f"🔊 Audio nivel 1 reproducido correctamente")
                else:
                    print(f"❌ Error: AlarmModule no pudo reproducir vadelante1")
                    
            elif level == 2:
                # Nivel 2: 7 segundos - comportamiento10s
                self.logger.info(f"🔊 Solicitando reproducción nivel 2: comportamiento10s")
                success = self.alarm_module.play_audio("comportamiento10s")
                if success:
                    print(f"🔊 Audio nivel 2 reproducido correctamente")
                else:
                    print(f"❌ Error: AlarmModule no pudo reproducir comportamiento10s")
                    
        except Exception as e:
            self.logger.error(f"❌ Error inesperado al reproducir audio: {e}")
            print(f"❌ Error al intentar reproducir audio nivel {level}: {str(e)}")
    
    def _draw_enhanced_visualization(self, frame, is_distracted):
        """Dibuja visualización mejorada con texto centrado en la parte inferior"""
        height, width = frame.shape[:2]
        
        # Color según estado
        if self.direction == "EXTREMO":
            color = (0, 0, 255)  # Rojo para giro extremo
            direction_text = "GIRO EXTREMO"
        elif is_distracted:
            intensity = 128 + int(127 * (self.distraction_counter / self.level2_threshold))
            color = (0, 0, min(255, intensity))
            direction_text = f"MIRANDO: {self.direction}"
        else:
            color = (0, 255, 0)
            direction_text = "MIRANDO: CENTRO"
        
        # Calcular posición centrada para el texto principal
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1.0
        text_thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(direction_text, font, text_scale, text_thickness)
        text_x = (width - text_width) // 2
        text_y = height - 50  # 50 pixels desde el borde inferior
        
        # Dibujar fondo semitransparente para el texto
        padding = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Dibujar el texto principal centrado
        cv2.putText(frame, direction_text, 
                   (text_x, text_y), 
                   font, text_scale, color, text_thickness)
        
        # Información de modo (esquina superior derecha)
        mode_text = f"MODO: {'NOCHE' if self.is_night_mode else 'DIA'}"
        cv2.putText(frame, mode_text, 
                   (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar nivel de alerta actual (centrado arriba)
        if self.current_alert_level > 0:
            alert_text = f"ALERTA NIVEL {self.current_alert_level}"
            (alert_width, alert_height), _ = cv2.getTextSize(alert_text, font, 0.8, 2)
            alert_x = (width - alert_width) // 2
            cv2.putText(frame, alert_text, 
                       (alert_x, 60), 
                       font, 0.8, (0, 0, 255), 2)
        
        # Barra de progreso centrada
        if is_distracted:
            bar_width = 400
            bar_height = 20
            bar_x = (width - bar_width) // 2
            bar_y = height - 120
            
            # Calcular progreso
            if self.distraction_counter < self.level1_threshold:
                progress = self.distraction_counter / self.level1_threshold
                target_time = self.config['level1_time']
                current_time = self.distraction_counter / self.config['camera_fps']
                level_text = "Nivel 1"
            else:
                progress = (self.distraction_counter - self.level1_threshold) / (self.level2_threshold - self.level1_threshold)
                target_time = self.config['level2_time'] - self.config['level1_time']
                current_time = (self.distraction_counter - self.level1_threshold) / self.config['camera_fps']
                level_text = "Nivel 2"
            
            # Texto de progreso centrado
            progress_text = f"{level_text}: {current_time:.1f}/{target_time:.1f} seg"
            (prog_width, prog_height), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            prog_x = (width - prog_width) // 2
            cv2.putText(frame, progress_text, 
                       (prog_x, bar_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Fondo de la barra
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Progreso actual
            filled_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         color, -1)
            
            # Marcador de nivel 1
            level1_x = bar_x + int(bar_width * (self.level1_threshold / self.level2_threshold))
            cv2.line(frame, (level1_x, bar_y - 5), (level1_x, bar_y + bar_height + 5), 
                    (255, 255, 0), 2)
        
        # Confianza (esquina superior izquierda)
        conf_text = f"Confianza: {self.detection_confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Contador de distracciones (esquina inferior derecha)
        count_color = (0, 0, 255) if len(self.distraction_times) >= 3 else (255, 255, 255)
        cv2.putText(frame, f"Distracciones: {len(self.distraction_times)}/3", 
                   (width - 200, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color, 1)
    
    def _detect_lighting_conditions(self, frame):
        """Detecta condiciones de iluminación para modo día/noche"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        self.light_level = np.mean(gray)
        
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.config['night_mode_threshold']
        
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
            print(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
    
    def get_config(self):
        """Retorna la configuración actual para el panel web"""
        return self.config.copy()
    
    def get_status(self):
        """Retorna el estado actual del detector"""
        fps = self.config['camera_fps']
        return {
            'direction': self.direction,
            'is_distracted': self.direction != "CENTRO",
            'distraction_counter': self.distraction_counter,
            'distraction_time': self.distraction_counter / fps,
            'current_alert_level': self.current_alert_level,
            'total_distractions': len(self.distraction_times),
            'confidence': self.detection_confidence,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level
        }