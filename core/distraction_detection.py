import time
import os
import pygame
import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque

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
        
        # 🆕 NUEVO: Cargar configuración externa (con fallbacks seguros)
        if CONFIG_AVAILABLE:
            # ===== CONFIGURACIÓN DESDE ARCHIVOS YAML =====
            self.config = {
                # Umbrales de rotación
                'rotation_threshold_day': get_config('distraction.rotation_threshold_day', 2.6),
                'rotation_threshold_night': get_config('distraction.rotation_threshold_night', 2.8),
                'extreme_rotation_threshold': get_config('distraction.extreme_rotation_threshold', 2.5),
                
                # Temporización de alertas (en segundos)
                'level1_time': get_config('distraction.level1_time', 3),
                'level2_time': get_config('distraction.level2_time', 5),
                
                # Sensibilidad y detección
                'visibility_threshold': get_config('distraction.visibility_threshold', 15),
                'frames_without_face_limit': get_config('distraction.frames_without_face_limit', 5),
                'confidence_threshold': get_config('distraction.confidence_threshold', 0.7),
                
                # Modo nocturno
                'night_mode_threshold': get_config('distraction.night_mode_threshold', 50),
                'enable_night_mode': get_config('distraction.enable_night_mode', True),
                
                # Buffer y ventanas de tiempo
                'prediction_buffer_size': get_config('distraction.prediction_buffer_size', 10),
                'distraction_window': get_config('distraction.distraction_window', 600),
                'min_frames_for_reset': get_config('distraction.min_frames_for_reset', 10),
                
                # Control de audio
                'audio_enabled': get_config('distraction.audio_enabled', True),
                'level1_volume': get_config('distraction.level1_volume', 0.8),
                'level2_volume': get_config('distraction.level2_volume', 1.0),
                
                # FPS de la cámara para cálculos
                'camera_fps': get_config('distraction.camera_fps', 4)
            }
            
            # 🆕 NUEVO: Configuración de GUI
            self.show_gui = has_gui()
            
            print(f"✅ DistractionDetector - Configuración cargada:")
            print(f"   - Umbral rotación día: {self.config['rotation_threshold_day']}")
            print(f"   - Umbral rotación noche: {self.config['rotation_threshold_night']}")
            print(f"   - Tiempo nivel 1: {self.config['level1_time']}s")
            print(f"   - Tiempo nivel 2: {self.config['level2_time']}s")
            print(f"   - GUI: {self.show_gui}")
            print(f"   - Audio: {self.config['audio_enabled']}")
        else:
            # ✅ FALLBACK: Configuración original si no hay sistema de config
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
            self.show_gui = True  # Default para compatibilidad
            print("⚠️ DistractionDetector usando configuración por defecto")
        
        # Calcular frames basados en configuración
        self.level1_threshold = int(self.config['level1_time'] * self.config['camera_fps'])
        self.level2_threshold = int(self.config['level2_time'] * self.config['camera_fps'])
        
        # Buffer para predicciones pasadas
        buffer_size = self.config['prediction_buffer_size']
        self.direction_buffer = deque(["CENTRO"] * buffer_size, maxlen=buffer_size)
        self.confidence_buffer = deque([1.0] * buffer_size, maxlen=buffer_size)
        
        # Estados previos para detección de pérdida de face
        self.last_valid_direction = "CENTRO"
        self.last_valid_confidence = 1.0
        self.frames_without_face = 0
        
        # Registro de distracciones
        self.distraction_times = []        
        self.distraction_counter = 0       
        self.current_alert_level = 0
        
        # Estado de modo (día/noche)
        self.is_night_mode = False
        self.light_level = 0
        
        # Inicializar el sistema de audio
        self._initialize_audio()
        
        # Para visualización
        self.direction = "CENTRO"          
        self.rotation_angle = 0            
        self.detection_confidence = 1.0    
        self.last_detection_time = 0       
        self.last_metrics = {}             
        
        # Para logs en modo headless
        self._last_log_time = 0
        
        print("=== Detector de Distracciones - Configuración Inicial ===")
        print(f"Tiempo Nivel 1: {self.config['level1_time']} segundos")
        print(f"Tiempo Nivel 2: {self.config['level2_time']} segundos")
        print(f"Modo nocturno: {'Habilitado' if self.config['enable_night_mode'] else 'Deshabilitado'}")
        print(f"Audio: {'Habilitado' if self.config['audio_enabled'] else 'Deshabilitado'}")
        
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
            self.level1_sound = None
            self.level2_sound = None
            return
            
        try:
            pygame.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            print("Sistema de audio pygame inicializado correctamente")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 🆕 NUEVO: Archivos de audio configurables
            if CONFIG_AVAILABLE:
                audio_level1 = get_config('audio.files.vadelante1', 'vadelante1.mp3')
                audio_level2 = get_config('audio.files.distraction', 'distraction.mp3')
            else:
                audio_level1 = 'vadelante1.mp3'
                audio_level2 = 'distraction.mp3'
            
            # Cargar audio nivel 1
            audio_path_1 = os.path.join(script_dir, "audio", audio_level1)
            if os.path.exists(audio_path_1):
                self.level1_sound = pygame.mixer.Sound(audio_path_1)
                self.level1_sound.set_volume(self.config['level1_volume'])
                print(f"✅ Audio nivel 1 cargado: {audio_path_1}")
            else:
                print(f"❌ ERROR: No se encontró {audio_level1}")
                self.level1_sound = None
            
            # Cargar audio nivel 2
            audio_path_2 = os.path.join(script_dir, "audio", audio_level2)
            if os.path.exists(audio_path_2):
                self.level2_sound = pygame.mixer.Sound(audio_path_2)
                self.level2_sound.set_volume(self.config['level2_volume'])
                print(f"✅ Audio nivel 2 cargado: {audio_path_2}")
            else:
                print(f"❌ ERROR: No se encontró {audio_level2}")
                self.level2_sound = None
                    
        except Exception as e:
            print(f"❌ ERROR al inicializar audio: {e}")
            self.level1_sound = None
            self.level2_sound = None
    
    def detect(self, landmarks, frame):
        """Detecta distracciones incluyendo giros extremos de cabeza"""
        # Detectar condiciones de iluminación si está habilitado
        if self.config['enable_night_mode']:
            self._detect_lighting_conditions(frame)
        
        # Primero verificar si tenemos landmarks válidos
        if landmarks is None or landmarks.num_parts == 0:
            self.frames_without_face += 1
            
            # Si perdemos la cara por más frames del límite configurado
            if self.frames_without_face > self.config['frames_without_face_limit']:
                if self.last_valid_direction != "CENTRO":
                    self.direction = self.last_valid_direction
                    self.detection_confidence = 0.5  
                else:
                    self.direction = "EXTREMO"
                    self.detection_confidence = 0.7
            
            return self._handle_distraction_timing(frame)
        
        # Si recuperamos la cara, resetear contador
        self.frames_without_face = 0
        
        # Verificar si es un giro extremo antes de calcular métricas
        is_extreme_rotation = self._check_extreme_rotation(landmarks, frame)
        
        if is_extreme_rotation:
            self.direction = "EXTREMO"
            self.detection_confidence = 0.8
            return self._handle_distraction_timing(frame)
        
        # Continuar con detección normal si no es giro extremo
        return self._detect_normal_rotation(landmarks, frame)
    
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
        """Detección normal cuando el rostro es visible"""
        # Seleccionar umbral según el modo
        current_threshold = (self.config['rotation_threshold_night'] if self.is_night_mode 
                           else self.config['rotation_threshold_day'])
        
        metrics = {}
        
        try:
            # Extraer puntos clave
            left_cheek = (landmarks.part(2).x, landmarks.part(2).y)
            right_cheek = (landmarks.part(14).x, landmarks.part(14).y)
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            
            # Calcular distancias nariz-mejillas
            dist_nose_left = distance.euclidean(nose_tip, left_cheek)
            dist_nose_right = distance.euclidean(nose_tip, right_cheek)
            
            if dist_nose_left > 0 and dist_nose_right > 0:
                cheek_ratio = dist_nose_right / dist_nose_left
            else:
                cheek_ratio = 1.0
                
            metrics['cheek_ratio'] = cheek_ratio
            
            # Determinar dirección (CORREGIDO: invertido para coincidir con vista)
            if cheek_ratio > current_threshold:
                self.direction = "DERECHA"    # Cambio aquí
            elif cheek_ratio < 1.0/current_threshold:
                self.direction = "IZQUIERDA"  # Cambio aquí
            else:
                self.direction = "CENTRO"
            
            self.detection_confidence = 1.0
            self.last_valid_direction = self.direction
            self.last_valid_confidence = self.detection_confidence
            
        except Exception as e:
            self.detection_confidence = 0.5
        
        return self._handle_distraction_timing(frame)
    
    def _handle_distraction_timing(self, frame):
        """Maneja el timing de distracciones y los dos niveles de alerta"""
        is_distracted = self.direction != "CENTRO"
        current_time = time.time()
        
        if is_distracted:
            self.distraction_counter += 1
            
            # Nivel 1
            if self.distraction_counter == self.level1_threshold and self.current_alert_level < 1:
                print(f"⚠️ NIVEL 1: Distracción detectada ({self.direction})")
                self._play_sound(1)
                self.current_alert_level = 1
            
            # Nivel 2
            elif self.distraction_counter == self.level2_threshold:
                print(f"🚨 NIVEL 2: Distracción prolongada ({self.direction})")
                self._play_sound(2)
                self.current_alert_level = 2
                
                # Registrar la distracción
                self.distraction_times.append(current_time)
                
                # Reiniciar contador después del nivel 2
                self.distraction_counter = 0
                self.current_alert_level = 0
        else:
            # Reiniciar contadores si vuelve al centro
            if self.distraction_counter > self.config['min_frames_for_reset']:
                print(f"Contador reiniciado: {self.distraction_counter} → 0")
            self.distraction_counter = 0
            self.current_alert_level = 0
        
        # Limpiar distracciones antiguas
        self.distraction_times = [t for t in self.distraction_times 
                                if t > current_time - self.config['distraction_window']]
        
        # Verificar múltiples distracciones
        multiple_distractions = len(self.distraction_times) >= 3
        
        # 🆕 NUEVO: Dibujar visualización solo si GUI está habilitada
        if self.show_gui:
            self._draw_enhanced_visualization(frame, is_distracted)
        else:
            # En modo headless, log periódico
            if current_time - self._last_log_time > 10:  # Log cada 10 segundos
                mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                print(f"📊 Distracción: {self.direction} | Confianza: {self.detection_confidence:.2f} | Nivel: {self.current_alert_level} | Modo: {mode_str} | Total: {len(self.distraction_times)}/3")
                self._last_log_time = current_time
        
        return is_distracted, multiple_distractions
    
    def _play_sound(self, level):
        """Reproduce el sonido correspondiente al nivel de alerta"""
        if not self.config['audio_enabled']:
            return
            
        try:
            pygame.mixer.stop()  
            
            if level == 1 and self.level1_sound:
                pygame.mixer.Channel(0).play(self.level1_sound)
                print(f"🔊 Reproduciendo alerta nivel 1")
            elif level == 2 and self.level2_sound:
                pygame.mixer.Channel(0).play(self.level2_sound)
                print(f"🔊 Reproduciendo alerta nivel 2")
                
        except Exception as e:
            print(f"❌ Error al reproducir sonido nivel {level}: {e}")
    
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