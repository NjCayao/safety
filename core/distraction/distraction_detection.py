import time
import os
import pygame
import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import logging

# Importar sistema de configuración
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

class DistractionDetector:
    def __init__(self):
        """Inicializa el detector de distracciones con configuración centralizada"""
        
        # Crear logger
        self.logger = logging.getLogger('DistractionDetector')
        
        # Cargar configuración
        if CONFIG_AVAILABLE:
            self.config = {
                'rotation_threshold_day': get_config('distraction.rotation_threshold_day', 2.6),
                'rotation_threshold_night': get_config('distraction.rotation_threshold_night', 2.8),
                'extreme_rotation_threshold': get_config('distraction.extreme_rotation_threshold', 2.5),
                'level1_time': get_config('distraction.level1_time', 3),
                'level2_time': get_config('distraction.level2_time', 7),
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
        else:
            self.config = {
                'rotation_threshold_day': 2.6,
                'rotation_threshold_night': 2.8,
                'extreme_rotation_threshold': 2.5,
                'level1_time': 3,
                'level2_time': 7,
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
        
        # Para compatibilidad con visualización
        self.level1_threshold = int(self.config['level1_time'] * self.config['camera_fps'])
        self.level2_threshold = int(self.config['level2_time'] * self.config['camera_fps'])
        
        # Inicializar buffers
        buffer_size = self.config['prediction_buffer_size']
        self.direction_buffer = deque(["CENTRO"] * buffer_size, maxlen=buffer_size)
        self.confidence_buffer = deque([1.0] * buffer_size, maxlen=buffer_size)
        
        # Estados
        self.last_valid_direction = "CENTRO"
        self.last_valid_confidence = 1.0
        self.frames_without_face = 0
        self.distraction_times = []
        self.distraction_counter = 0
        self.current_alert_level = 0
        self.is_night_mode = False
        self.light_level = 100
        
        # Variables para tiempo real
        self.distraction_start_time = None
        self.level1_triggered = False
        self.level2_triggered = False
        self.time_in_center = 0
        self.last_center_time = None
        
        # Variables para audio
        self.alarm_module = None
        self.audio_ready = False
        
        # Variables para visualización
        self.direction = "CENTRO"          
        self.rotation_angle = 0            
        self.detection_confidence = 1.0    
        self.last_detection_time = 0       
        self.last_metrics = {}             
        self._last_log_time = 0
        
        # Para dibujar líneas faciales
        self.last_landmarks = None
        self.last_face_rect = None
        
        # Info de última detección
        self.last_detection_info = {}
        
        self.logger.info("Detector de Distracciones inicializado (basado en tiempo)")
        
    def update_config(self, new_config):
        """Actualiza la configuración desde el panel web"""
        self.config.update(new_config)
        self.level1_threshold = int(self.config['level1_time'] * self.config['camera_fps'])
        self.level2_threshold = int(self.config['level2_time'] * self.config['camera_fps'])

    def set_alarm_module(self, alarm_module):
        """Configura la referencia al AlarmModule"""
        self.alarm_module = alarm_module
        self.logger.info("AlarmModule configurado en DistractionDetector")
    
    def detect(self, landmarks, frame):
        """Detecta distracciones enfocándose SOLO en giros extremos"""
        
        # Guardar landmarks para dibujar
        self.last_landmarks = landmarks
        
        # Detectar condiciones de iluminación si está habilitado
        if self.config['enable_night_mode'] and frame is not None:
            self._detect_lighting_conditions(frame)
        
        # Primero verificar si tenemos landmarks válidos
        if landmarks is None or landmarks.num_parts == 0:
            self.frames_without_face += 1
            
            # Lógica mejorada para distinguir entre ausencia y giro extremo
            if self.frames_without_face <= 2:
                # Muy pocos frames: mantener último estado
                if not hasattr(self, 'last_known_direction'):
                    self.last_known_direction = "CENTRO"
                self.direction = self.last_known_direction
                self.detection_confidence = max(0.3, self.detection_confidence - 0.1)
            elif self.frames_without_face <= 10:
                # Pérdida breve: mostrar sin rostro temporalmente
                self.direction = "SIN ROSTRO"
                self.detection_confidence = 0.3
            elif self.frames_without_face <= 30:  # ~7.5 segundos
                # Pérdida mediana: verificar si había indicios de giro antes
                if hasattr(self, 'last_detection_info') and self.last_detection_info.get('ear_visibility', 0) > 0.4:
                    # Había visibilidad de perfil antes: probablemente giro extremo
                    self.direction = "EXTREMO"
                    self.detection_confidence = 0.6
                else:
                    # No había perfil visible: mantener sin rostro
                    self.direction = "SIN ROSTRO"
                    self.detection_confidence = 0.2
            else:
                # Pérdida muy prolongada: asumir ausencia real
                self.direction = "AUSENTE"
                self.detection_confidence = 0.1
                # Resetear para no contar como distracción
                if self.distraction_start_time:
                    self.distraction_start_time = None
                    self.current_alert_level = 0
            
            return self._handle_distraction_timing(frame)
        
        # Si recuperamos la cara, resetear contador y actualizar dirección conocida
        self.frames_without_face = 0
        self.last_known_direction = "CENTRO"  # Actualizar última dirección conocida
        
        # SOLO verificar si es un giro extremo
        is_extreme_rotation = self._check_extreme_rotation(landmarks, frame)
        
        if is_extreme_rotation:
            self.direction = "EXTREMO"
            self.detection_confidence = 0.9
        else:
            # Si no es extremo, está mirando "normal"
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
            
            # Guardar rectángulo de la cara
            self.last_face_rect = (leftmost, topmost, rightmost, bottommost)
            
            # NUEVA LÓGICA: Detectar visibilidad de oreja
            # Los puntos 0-3 y 13-16 del contorno corresponden a las zonas cerca de las orejas
            left_ear_region = jaw_points[0:4]   # Cerca de oreja izquierda
            right_ear_region = jaw_points[13:17] # Cerca de oreja derecha
            
            # Calcular cuánto del perfil lateral es visible
            face_center_x = (leftmost + rightmost) / 2
            
            # Distancia de los puntos de oreja al centro
            left_ear_distance = abs(left_ear_region[0][0] - face_center_x)
            right_ear_distance = abs(right_ear_region[-1][0] - face_center_x)
            
            # Si un lado está muy extendido, significa que vemos más perfil (oreja visible)
            ear_visibility_ratio = max(left_ear_distance, right_ear_distance) / face_width if face_width > 0 else 0
            
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
            nose_offset = abs(nose_x - face_center_x) / face_width if face_width > 0 else 0
            
            # CRITERIOS MEJORADOS para giro extremo
            # Debe verse parte del perfil (oreja) para confirmar que es giro y no ausencia
            is_extreme = (
                (aspect_ratio < 0.5 or eye_visibility_ratio < 0.5 or nose_offset > 0.4) and
                ear_visibility_ratio > 0.45  # Vemos bastante del perfil lateral
            )
            
            # Guardar información de detección
            self.last_detection_info = {
                'aspect_ratio': aspect_ratio,
                'ear_visibility': ear_visibility_ratio,
                'eye_visibility': eye_visibility_ratio,
                'nose_offset': nose_offset
            }
            
            if is_extreme:
                # Determinar dirección del giro extremo
                if nose_x < face_center_x:
                    self.last_valid_direction = "IZQUIERDA"
                else:
                    self.last_valid_direction = "DERECHA"
                self.last_known_direction = "EXTREMO"  # Guardar que hubo giro extremo
            
            return is_extreme
            
        except Exception as e:
            return True
    
    def _handle_distraction_timing(self, frame):
        """Maneja el timing de distracciones usando TIEMPO REAL"""
        
        # Considerar distracción tanto en EXTREMO como cuando pierde rostro por mucho tiempo
        is_distracted = (self.direction == "EXTREMO")
        current_time = time.time()
        
        # IMPORTANTE: Limpiar distracciones antiguas ANTES de procesar
        self._clean_old_distractions()
        
        if is_distracted:
            # Iniciar timer si es nueva distracción
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                self.level1_triggered = False
                self.level2_triggered = False
                self.last_center_time = None
            
            # Calcular tiempo transcurrido
            elapsed_time = current_time - self.distraction_start_time
            
            # Nivel 1: A los 3 segundos exactos
            if elapsed_time >= self.config['level1_time'] and not self.level1_triggered:
                print(f"⚠️ NIVEL 1: Giro extremo detectado ({elapsed_time:.1f} segundos)")
                self._play_sound(1)
                self.current_alert_level = 1
                self.level1_triggered = True
            
            # Nivel 2: A los 7 segundos exactos
            elif elapsed_time >= self.config['level2_time'] and not self.level2_triggered:
                print(f"🚨 NIVEL 2: Giro extremo prolongado ({elapsed_time:.1f} segundos)")
                self._play_sound(2)
                self.current_alert_level = 2
                self.level2_triggered = True
                
                # Registrar la distracción
                self.distraction_times.append(current_time)
                print(f"📊 Giro extremo #{len(self.distraction_times)} registrado")
            
            # Para la visualización, convertir tiempo a frames aproximados
            self.distraction_counter = int(elapsed_time * self.config['camera_fps'])
            
        else:
            # Está en centro
            if self.last_center_time is None:
                self.last_center_time = current_time
            
            # Si estaba distraído y ahora está en centro
            if self.distraction_start_time is not None:
                elapsed = current_time - self.distraction_start_time
                time_in_center = current_time - self.last_center_time
                
                # Requiere estar 0.75 segundos en centro para resetear
                if time_in_center >= 0.75:
                    if elapsed >= 1.0:
                        print(f"✅ Volvió a posición normal tras {elapsed:.1f}s")
                        if self.current_alert_level >= 2:
                            print(f"   → Evento registrado. Total: {len(self.distraction_times)}/3")
                    
                    # Resetear
                    self.distraction_start_time = None
                    self.distraction_counter = 0
                    self.current_alert_level = 0
                    self.last_center_time = None
        
        # Verificar múltiples distracciones
        multiple_distractions = len(self.distraction_times) >= 3
        
        # Dibujar visualización solo si GUI está habilitada
        if self.show_gui and frame is not None:
            self._draw_enhanced_visualization(frame, is_distracted)
            # Dibujar líneas faciales
            self._draw_face_lines(frame)
        
        return is_distracted, multiple_distractions
    
    def _clean_old_distractions(self):
        """Elimina distracciones fuera de la ventana temporal"""
        current_time = time.time()
        window = self.config['distraction_window']
        
        # Filtrar solo las distracciones dentro de la ventana
        old_count = len(self.distraction_times)
        self.distraction_times = [t for t in self.distraction_times 
                                if current_time - t < window]
        
        # Si se eliminaron distracciones, informar
        if old_count > len(self.distraction_times):
            removed = old_count - len(self.distraction_times)
            print(f"🔄 Se eliminaron {removed} distracciones antiguas (más de {window//60} minutos)")
    
    def _play_sound(self, level):
        """Reproduce el sonido correspondiente al nivel de alerta usando AlarmModule"""
        if not self.config['audio_enabled']:
            return
            
        try:
            if not hasattr(self, 'alarm_module') or self.alarm_module is None:
                self.logger.warning("AlarmModule no está configurado")
                return
            
            if level == 1:
                self.alarm_module.stop_audio()
                self.logger.info(f"🔊 Reproduciendo nivel 1: vadelante1")
                success = self.alarm_module.play_audio("vadelante1")
                if success:
                    print(f"🔊 Audio nivel 1 reproducido correctamente")
                    
            elif level == 2:
                self.logger.info(f"🔊 Reproduciendo nivel 2: comportamiento10s")
                success = self.alarm_module.play_audio("comportamiento10s")
                if success:
                    print(f"🔊 Audio nivel 2 reproducido correctamente")
                    
        except Exception as e:
            self.logger.error(f"Error al reproducir audio: {e}")
    
    def _draw_face_lines(self, frame):
        """Dibuja solo puntos en las características faciales clave"""
        if self.last_landmarks is None:
            return
        
        try:
            # Color para los puntos
            point_color = (0, 165, 255)  # Naranja
            point_radius = 4
            
            # Lista de puntos clave a dibujar
            key_points = [
                36,  # Ojo izquierdo externo
                39,  # Ojo izquierdo interno
                42,  # Ojo derecho interno
                45,  # Ojo derecho externo
                30,  # Punta de la nariz
                48,  # Comisura izquierda de la boca
                51,  # Centro superior de la boca
                54,  # Comisura derecha de la boca
                57,  # Centro inferior de la boca
                3,   # Mejilla izquierda
                13,  # Mejilla derecha
                8    # Mentón
            ]
            
            # Dibujar solo los puntos
            for point_idx in key_points:
                x = self.last_landmarks.part(point_idx).x
                y = self.last_landmarks.part(point_idx).y
                cv2.circle(frame, (x, y), point_radius, point_color, -1)
            
        except Exception as e:
            pass
    
    def _draw_enhanced_visualization(self, frame, is_distracted):
        """Dibuja visualización mejorada con texto centrado en la parte inferior"""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        
        # Color según estado
        if self.direction == "EXTREMO":
            color = (0, 0, 255)
            direction_text = "GIRO EXTREMO"
        elif self.direction == "SIN ROSTRO":
            color = (255, 165, 0)  # Naranja
            direction_text = "ROSTRO NO DETECTADO"
        elif self.direction == "AUSENTE":
            color = (128, 128, 128)  # Gris
            direction_text = "CONDUCTOR AUSENTE"
        else:
            color = (0, 255, 0)
            direction_text = "MIRANDO: CENTRO"
        
        # Calcular posición centrada para el texto principal
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1.0
        text_thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(direction_text, font, text_scale, text_thickness)
        text_x = (width - text_width) // 2
        text_y = height - 50
        
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
        if is_distracted and self.distraction_start_time:
            bar_width = 400
            bar_height = 20
            bar_x = (width - bar_width) // 2
            bar_y = height - 120
            
            # Calcular tiempo real transcurrido
            elapsed_time = time.time() - self.distraction_start_time
            
            if elapsed_time < self.config['level1_time']:
                # Hacia nivel 1
                progress = elapsed_time / self.config['level1_time']
                target_time = self.config['level1_time']
                level_text = "Nivel 1"
            else:
                # Hacia nivel 2
                progress = elapsed_time / self.config['level2_time']
                target_time = self.config['level2_time']
                level_text = "Nivel 2"
            
            # Texto de progreso centrado
            progress_text = f"{level_text}: {elapsed_time:.1f}/{target_time:.1f} seg"
            (prog_width, prog_height), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            prog_x = (width - prog_width) // 2
            cv2.putText(frame, progress_text, 
                       (prog_x, bar_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Fondo de la barra
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Progreso actual
            filled_width = int(bar_width * min(1.0, progress))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         color, -1)
            
            # Marcador de nivel 1
            level1_x = bar_x + int(bar_width * (self.config['level1_time'] / self.config['level2_time']))
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
        try:
            if frame is None or frame.size == 0:
                self.light_level = 100
                self.is_night_mode = False
                return
                
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            self.light_level = np.mean(gray)
            
            if self.light_level < 5:
                self.light_level = 100
                
            previous_mode = self.is_night_mode
            self.is_night_mode = self.light_level < self.config['night_mode_threshold']
            
            if previous_mode != self.is_night_mode:
                mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
                print(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
                
        except Exception as e:
            self.light_level = 100
            self.is_night_mode = False
    
    def get_config(self):
        """Retorna la configuración actual para el panel web"""
        return self.config.copy()
    
    def get_status(self):
        """Retorna el estado actual del detector"""
        # Calcular tiempo de distracción real
        if self.distraction_start_time:
            distraction_time = time.time() - self.distraction_start_time
        else:
            distraction_time = 0
        
        return {
            'direction': self.direction,
            'is_distracted': self.direction != "CENTRO",
            'distraction_counter': self.distraction_counter,
            'distraction_time': distraction_time,
            'current_alert_level': self.current_alert_level,
            'total_distractions': len(self.distraction_times),
            'confidence': self.detection_confidence,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'level1_time': self.config['level1_time'],
            'level2_time': self.config['level2_time']
        }