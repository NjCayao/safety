import os
import time
import cv2
import numpy as np
from collections import deque
from scipy.spatial import distance
import pygame

# 🆕 NUEVO: Importar sistema de configuración
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Sistema de configuración no disponible para BostezosDetector, usando valores por defecto")

class BostezosDetector:
    def __init__(self):
        """Inicializa el detector de bostezos"""
        
        # 🆕 NUEVO: Cargar configuración externa (con fallbacks seguros)
        if CONFIG_AVAILABLE:
            # === CONFIGURACIÓN DESDE ARCHIVOS YAML ===
            
            # Configuración de detección
            self.config = {
                'YAWN_THRESHOLD': get_config('yawn.mouth_threshold', 0.7),
                'YAWN_DURATION_THRESHOLD': get_config('yawn.duration_threshold', 2.5),
                'WINDOW_SIZE': get_config('yawn.window_size', 600),
                'FRAMES_TO_CONFIRM': get_config('yawn.frames_to_confirm', 3),
                
                # Configuración de alertas
                'ALERT_COOLDOWN': get_config('yawn.alert_cooldown', 5.0),
                'MAX_YAWNS_BEFORE_ALERT': get_config('yawn.max_yawns_before_alert', 3),
                'REPORT_DELAY': get_config('yawn.report_delay', 2.0),
                
                # Configuración de modo nocturno/infrarrojo
                'ENABLE_NIGHT_MODE': get_config('yawn.enable_night_mode', True),
                'NIGHT_MODE_THRESHOLD': get_config('yawn.night_mode_threshold', 50),
                'NIGHT_ADJUSTMENT': get_config('yawn.night_adjustment', 0.05),
                
                # Configuración de calibración
                'ENABLE_AUTO_CALIBRATION': get_config('yawn.enable_auto_calibration', True),
                'CALIBRATION_FRAMES': get_config('yawn.calibration_frames', 60),
                'CALIBRATION_FACTOR': get_config('yawn.calibration_factor', 0.4),
                
                # Configuración de audio
                'ENABLE_SOUNDS': get_config('yawn.enable_sounds', True),
                'AUDIO_FREQUENCY': get_config('yawn.audio_frequency', 44100),
                'AUDIO_CHANNELS': get_config('yawn.audio_channels', 2),
                'AUDIO_BUFFER': get_config('yawn.audio_buffer', 2048),
                
                # Configuración visual
                'SHOW_DEBUG_INFO': get_config('yawn.show_debug_info', True),
                'SHOW_MOUTH_CONTOUR': get_config('yawn.show_mouth_contour', True),
                'SHOW_PROGRESS_BAR': get_config('yawn.show_progress_bar', True),
                'SHOW_LIGHT_LEVEL': get_config('yawn.show_light_level', True),
                'SHOW_MAR_VALUE': get_config('yawn.show_mar_value', True),
                'SHOW_MODE_INDICATOR': get_config('yawn.show_mode_indicator', True),
            }
            
            # 🆕 NUEVO: Configuración de GUI
            self.show_gui = has_gui()
            
            print(f"✅ BostezosDetector - Configuración cargada:")
            print(f"   - Umbral boca: {self.config['YAWN_THRESHOLD']}")
            print(f"   - Duración mínima: {self.config['YAWN_DURATION_THRESHOLD']}s")
            print(f"   - GUI: {self.show_gui}")
            print(f"   - Audio: {self.config['ENABLE_SOUNDS']}")
        else:
            # ✅ FALLBACK: Configuración original si no hay sistema de config
            self.config = {
                'YAWN_THRESHOLD': 0.7,
                'YAWN_DURATION_THRESHOLD': 2.5,
                'WINDOW_SIZE': 600,
                'FRAMES_TO_CONFIRM': 3,
                'ALERT_COOLDOWN': 5.0,
                'MAX_YAWNS_BEFORE_ALERT': 3,
                'REPORT_DELAY': 2.0,
                'ENABLE_NIGHT_MODE': True,
                'NIGHT_MODE_THRESHOLD': 50,
                'NIGHT_ADJUSTMENT': 0.05,
                'ENABLE_AUTO_CALIBRATION': True,
                'CALIBRATION_FRAMES': 60,
                'CALIBRATION_FACTOR': 0.4,
                'ENABLE_SOUNDS': True,
                'AUDIO_FREQUENCY': 44100,
                'AUDIO_CHANNELS': 2,
                'AUDIO_BUFFER': 2048,
                'SHOW_DEBUG_INFO': True,
                'SHOW_MOUTH_CONTOUR': True,
                'SHOW_PROGRESS_BAR': True,
                'SHOW_LIGHT_LEVEL': True,
                'SHOW_MAR_VALUE': True,
                'SHOW_MODE_INDICATOR': True,
            }
            self.show_gui = True  # Default para compatibilidad
            print("⚠️ BostezosDetector usando configuración por defecto")
        
        # 🆕 NUEVO: Archivos de audio configurables
        if CONFIG_AVAILABLE:
            self.audio_files = {
                'yawn_1': get_config('audio.files.bostezo1', 'bostezo1.mp3'),
                'yawn_2': get_config('audio.files.bostezo2', 'bostezo2.mp3'),
                'yawn_3': get_config('audio.files.bostezo3', 'bostezo3.mp3'),
                'fallback': get_config('audio.files.fallback', 'alarma.mp3')
            }
        else:
            self.audio_files = {
                'yawn_1': 'bostezo1.mp3',
                'yawn_2': 'bostezo2.mp3',
                'yawn_3': 'bostezo3.mp3',
                'fallback': 'alarma.mp3'
            }
        
        # 🆕 NUEVO: Colores configurables para visualización (BGR)
        if CONFIG_AVAILABLE:
            # Podríamos hacer estos configurables también, pero por ahora usar defaults
            pass
        
        self.colors = {
            'mouth_normal': (0, 255, 0),      # Verde
            'mouth_yawning': (0, 0, 255),     # Rojo
            'text_normal': (255, 255, 255),   # Blanco
            'text_warning': (0, 165, 255),    # Naranja
            'text_critical': (0, 0, 255),     # Rojo
            'overlay_fatigue': (0, 0, 100),   # Rojo oscuro
            'progress_bar_bg': (100, 100, 100), # Gris
            'progress_bar_fill': (0, 165, 255), # Naranja
        }
        
        # --- Posiciones de texto en pantalla ---
        self.text_positions = {
            'mouth_status': (10, 0.4),       # x, ratio vertical (0.4 = 40% desde arriba)
            'mar_value': (10, 0.45),         # 45% desde arriba
            'light_level': (10, 0.5),        # 50% desde arriba (centro)
            'duration': (10, 0.55),          # 55% desde arriba
            'progress_bar': (10, 0.6),       # 60% desde arriba
            'yawn_count': (10, 0.65),        # 65% desde arriba
        }
        
        # ===== VARIABLES DE ESTADO INTERNO (IGUALES QUE ANTES) =====
        
        # Estado de detección
        self.yawn_times = deque()
        self.yawn_in_progress = False
        self.yawn_start_time = None
        self.last_alert_time = 0
        self.third_yawn_alerted = False
        self.report_sent_time = 0
        
        # Contadores de suavizado
        self.yawn_counter = 0
        self.normal_counter = 0
        
        # Estado de modo nocturno
        self.is_night_mode = False
        self.light_level = 0
        
        # Variables de calibración
        self.calibration_frame_count = 0
        self.min_mar_observed = 1.0
        self.max_mar_observed = 0.0
        
        # Sistema de audio
        self.yawn_sounds = {}
        
        # Inicializar sistema
        self._initialize_audio()
        self._print_config()
    
    def _print_config(self):
        """Imprime la configuración actual"""
        print("=== Inicializando Detector de Bostezos ===")
        print(f"Umbral de apertura: {self.config['YAWN_THRESHOLD']}")
        print(f"Duración mínima: {self.config['YAWN_DURATION_THRESHOLD']} segundos")
        print(f"Modo nocturno: {'Auto-detección habilitada' if self.config['ENABLE_NIGHT_MODE'] else 'Deshabilitado'}")
        print(f"Umbral de luz para modo nocturno: {self.config['NIGHT_MODE_THRESHOLD']}")
        print(f"Ajuste de umbral en modo nocturno: {self.config['NIGHT_ADJUSTMENT']}")
        print(f"Compatible con cámaras infrarrojas: Sí")
        print(f"GUI habilitada: {self.show_gui}")
    
    def update_config(self, new_config):
        """Actualiza la configuración desde el panel web"""
        self.config.update(new_config)
        self._print_config()
        print("Configuración actualizada desde panel web")
    
    def get_config(self):
        """Devuelve la configuración actual para el panel web"""
        return self.config.copy()
    
    def get_status(self):
        """Devuelve el estado actual para el panel web"""
        return {
            'yawn_count': len(self.yawn_times),
            'is_yawning': self.yawn_in_progress,
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'last_yawn_time': self.yawn_times[-1] if self.yawn_times else None,
            'calibration_complete': self.calibration_frame_count >= self.config['CALIBRATION_FRAMES'],
            'mar_threshold': self.config['YAWN_THRESHOLD'],
        }
    
    def _initialize_audio(self):
        """Inicializa el sistema de audio"""
        try:
            pygame.init()
            pygame.mixer.init(
                frequency=self.config['AUDIO_FREQUENCY'], 
                size=-16, 
                channels=self.config['AUDIO_CHANNELS'], 
                buffer=self.config['AUDIO_BUFFER']
            )
            print("Sistema de audio inicializado correctamente")
            
            # Cargar sonidos progresivos de bostezos
            self.yawn_sounds = {
                1: self._load_audio_file(self.audio_files['yawn_1']),
                2: self._load_audio_file(self.audio_files['yawn_2']),
                3: self._load_audio_file(self.audio_files['yawn_3'])
            }
            
            # Verificar que todos los archivos se cargaron correctamente
            for level, sound in self.yawn_sounds.items():
                if sound is None:
                    print(f"⚠️ Advertencia: No se pudo cargar {self.audio_files[f'yawn_{level}']}")
                    # Fallback a un archivo genérico si existe
                    self.yawn_sounds[level] = self._load_audio_file(self.audio_files['fallback'])
                
        except Exception as e:
            print(f"❌ ERROR al inicializar audio: {e}")
            import traceback
            traceback.print_exc()
            self.yawn_sounds = {}
    
    def _load_audio_file(self, filename):
        """Carga un archivo de audio"""
        try:
            path = os.path.join("audio", filename)
            if os.path.exists(path):
                print(f"Archivo de audio cargado: {filename}")
                return pygame.mixer.Sound(path)
            print(f"Advertencia: Archivo de audio no encontrado - {filename}")
            return None
        except Exception as e:
            print(f"Error cargando {filename}: {e}")
            return None
    
    def _play_yawn_sound(self, yawn_number):
        """Reproduce el sonido correspondiente al número de bostezo"""
        if not self.config['ENABLE_SOUNDS']:
            return
            
        try:
            sound = self.yawn_sounds.get(yawn_number)
            
            if sound is not None:
                pygame.mixer.stop()
                pygame.mixer.Channel(0).play(sound)
                print(f"Reproduciendo {self.audio_files[f'yawn_{yawn_number}']}")
                return True
            else:
                print(f"⚠️ No se pudo reproducir sonido para bostezo #{yawn_number}")
                return False
        except Exception as e:
            print(f"❌ Error al reproducir sonido: {e}")
            return False
    
    def _detect_lighting_conditions(self, frame):
        """Detecta las condiciones de iluminación y determina si es modo nocturno"""
        if not self.config['ENABLE_NIGHT_MODE']:
            return False
            
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        self.light_level = np.mean(gray)
        
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.config['NIGHT_MODE_THRESHOLD']
        
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO (IR)" if self.is_night_mode else "DIURNO"
            print(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
            if self.is_night_mode:
                print(f"Ajustando umbral para modo nocturno: -{self.config['NIGHT_ADJUSTMENT']}")
            else:
                print(f"Restaurando umbral para modo diurno")
                
        return self.is_night_mode
    
    def _enhance_ir_image(self, frame):
        """Mejora la imagen para cámaras infrarrojas"""
        if self.is_night_mode:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            enhanced = cv2.equalizeHist(gray)
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            if len(frame.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
        else:
            return frame
    
    def detect(self, landmarks, frame):
        """Detecta bostezos basados en la apertura de la boca"""
        # Detectar condiciones de iluminación
        self._detect_lighting_conditions(frame)
        
        # Mejorar imagen si es modo nocturno/IR
        if self.is_night_mode:
            display_frame = self._enhance_ir_image(frame.copy())
        else:
            display_frame = frame
        
        # Obtener umbral ajustado según modo
        current_threshold = self.config['YAWN_THRESHOLD']
        if self.is_night_mode:
            current_threshold -= self.config['NIGHT_ADJUSTMENT']
        
        # Obtener tiempo actual
        current_time = time.time()
        
        # Verificar si necesitamos mantener el estado de múltiples bostezos
        maintain_multiple_yawns = False
        if hasattr(self, 'report_sent_time') and self.report_sent_time > 0:
            if current_time - self.report_sent_time < self.config['REPORT_DELAY']:
                maintain_multiple_yawns = True
            else:
                self.report_sent_time = 0
                if len(self.yawn_times) >= self.config['MAX_YAWNS_BEFORE_ALERT']:
                    print("⚠️ REINICIANDO CONTADOR DE BOSTEZOS (post-reporte)")
                    self.yawn_times.clear()
                    self.third_yawn_alerted = False
        
        # Extraer puntos de la boca
        mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        
        # Calcular MAR (Mouth Aspect Ratio)
        top_lip = mouth_points[3:7]
        bottom_lip = mouth_points[9:13]
        
        top_mean = np.mean([p[1] for p in top_lip])
        bottom_mean = np.mean([p[1] for p in bottom_lip])
        
        mouth_height = bottom_mean - top_mean
        mouth_width = distance.euclidean(mouth_points[0], mouth_points[6])
        
        mar = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Calibración automática
        if self.config['ENABLE_AUTO_CALIBRATION'] and self.calibration_frame_count < self.config['CALIBRATION_FRAMES']:
            self.calibration_frame_count += 1
            self.min_mar_observed = min(self.min_mar_observed, mar)
            self.max_mar_observed = max(self.max_mar_observed, mar)
            
            if self.calibration_frame_count == self.config['CALIBRATION_FRAMES']:
                range_mar = self.max_mar_observed - self.min_mar_observed
                if range_mar > 0.1:
                    new_threshold = self.min_mar_observed + (range_mar * self.config['CALIBRATION_FACTOR'])
                    print(f"Calibración completada. Nuevo umbral MAR: {new_threshold:.2f}")
                    self.config['YAWN_THRESHOLD'] = new_threshold
                else:
                    print("Rango MAR insuficiente para calibración automática")
        
        # Determinar si hay un bostezo
        current_yawn = mar > current_threshold
        
        # Suavizar detección
        if current_yawn:
            self.yawn_counter += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            self.yawn_counter = 0
        
        confirmed_yawn = self.yawn_counter >= self.config['FRAMES_TO_CONFIRM']
        confirmed_normal = self.normal_counter >= self.config['FRAMES_TO_CONFIRM']
        
        # Lógica de detección
        is_yawning = False
        multiple_yawns = False
        
        if confirmed_yawn and not self.yawn_in_progress:
            self.yawn_in_progress = True
            self.yawn_start_time = current_time
            print(f"Inicio de bostezo detectado (MAR: {mar:.2f})")
            
        elif confirmed_normal and self.yawn_in_progress:
            self.yawn_in_progress = False
            yawn_duration = current_time - self.yawn_start_time
            
            if yawn_duration >= self.config['YAWN_DURATION_THRESHOLD']:
                self.yawn_times.append(current_time)
                
                # Limpiar bostezos antiguos
                while self.yawn_times and (current_time - self.yawn_times[0] > self.config['WINDOW_SIZE']):
                    self.yawn_times.popleft()
                    print(f">>> Bostezo antiguo eliminado. Nuevo contador: {len(self.yawn_times)}/{self.config['MAX_YAWNS_BEFORE_ALERT']}")
                    if len(self.yawn_times) < self.config['MAX_YAWNS_BEFORE_ALERT']:
                        self.third_yawn_alerted = False
                
                current_count = len(self.yawn_times)
                print(f"Bostezo completo registrado: {yawn_duration:.1f} segundos")
                print(f">>> Estado del contador: {current_count}/{self.config['MAX_YAWNS_BEFORE_ALERT']} bostezos")
                
                # Reproducir sonido
                if current_time - self.last_alert_time > self.config['ALERT_COOLDOWN']:
                    self.last_alert_time = current_time
                    
                    if current_count == 1:
                        print("¡PRIMER BOSTEZO REGISTRADO!")
                        self._play_yawn_sound(1)
                    elif current_count == 2:
                        print("¡SEGUNDO BOSTEZO REGISTRADO!")
                        self._play_yawn_sound(2)
                    elif current_count >= 3:
                        print("¡TERCER BOSTEZO O MÁS REGISTRADO!")
                        self._play_yawn_sound(3)
                
                # Verificar múltiples bostezos
                if len(self.yawn_times) >= self.config['MAX_YAWNS_BEFORE_ALERT']:
                    if len(self.yawn_times) == self.config['MAX_YAWNS_BEFORE_ALERT'] and not self.third_yawn_alerted:
                        print(f"¡{self.config['MAX_YAWNS_BEFORE_ALERT']} BOSTEZOS DETECTADOS EN {self.config['WINDOW_SIZE']//60} MINUTOS! - ALERTA DE FATIGA")
                        self.third_yawn_alerted = True
                        self.report_sent_time = current_time
                        print(">>> MULTIPLE_YAWNS = True (main_system generará el reporte)")
            else:
                print(f"Bostezo ignorado (duración insuficiente: {yawn_duration:.1f}s)")
        
        is_yawning = self.yawn_in_progress
        
        if maintain_multiple_yawns:
            multiple_yawns = True
        else:
            multiple_yawns = len(self.yawn_times) >= self.config['MAX_YAWNS_BEFORE_ALERT']
        
        # 🆕 NUEVO: Dibujar información solo si GUI está habilitada
        if self.show_gui:
            display_frame = self._draw_yawn_info(display_frame, mouth_points, mar, current_threshold, is_yawning)
        else:
            # En modo headless, log periódico
            if hasattr(self, '_last_log_time'):
                if current_time - self._last_log_time > 10:  # Log cada 10 segundos
                    status = "BOSTEZANDO" if is_yawning else "Normal"
                    mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                    print(f"📊 Bostezos: {status} | MAR: {mar:.2f} | Umbral: {current_threshold:.2f} | Modo: {mode_str} | Contador: {len(self.yawn_times)}/{self.config['MAX_YAWNS_BEFORE_ALERT']}")
                    self._last_log_time = current_time
            else:
                self._last_log_time = current_time
        
        return is_yawning, multiple_yawns
    
    def _draw_yawn_info(self, frame, mouth_points, mar, threshold, is_yawning):
        """Dibuja información visual sobre la detección de bostezos"""
        h, w = frame.shape[:2]
        
        # Dibujar contorno de la boca
        if self.config['SHOW_MOUTH_CONTOUR']:
            hull = cv2.convexHull(np.array(mouth_points))
            color = self.colors['mouth_yawning'] if is_yawning else self.colors['mouth_normal']
            cv2.drawContours(frame, [hull], -1, color, 1)
        
        # Estado actual
        status = "BOSTEZANDO" if is_yawning else "Normal"
        color = self.colors['text_critical'] if is_yawning else self.colors['mouth_normal']
        y_pos = int(h * self.text_positions['mouth_status'][1])
        cv2.putText(frame, f"Boca: {status}", (self.text_positions['mouth_status'][0], y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Mostrar MAR
        if self.config['SHOW_MAR_VALUE']:
            mode_str = "NOCHE (IR)" if self.is_night_mode else "DÍA"
            y_pos = int(h * self.text_positions['mar_value'][1])
            cv2.putText(frame, f"MAR: {mar:.2f} (Umbral: {threshold:.2f}, Modo: {mode_str})", 
                    (self.text_positions['mar_value'][0], y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_critical'], 1)
        
        # Nivel de luz
        if self.config['SHOW_LIGHT_LEVEL']:
            y_pos = int(h * self.text_positions['light_level'][1])
            cv2.putText(frame, f"Nivel luz: {self.light_level:.1f}", 
                    (self.text_positions['light_level'][0], y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_normal'], 1)
        
        # Duración y barra de progreso
        if is_yawning and self.yawn_start_time and self.config['SHOW_PROGRESS_BAR']:
            duration = time.time() - self.yawn_start_time
            progress = min(duration / self.config['YAWN_DURATION_THRESHOLD'], 1.0) * 100
            
            y_pos = int(h * self.text_positions['duration'][1])
            cv2.putText(frame, f"Duración: {duration:.1f}s / {self.config['YAWN_DURATION_THRESHOLD']}s ({progress:.0f}%)", 
                    (self.text_positions['duration'][0], y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_warning'], 2)
            
            # Barra de progreso
            bar_width = 200
            filled_width = int(bar_width * progress / 100)
            y_pos = int(h * self.text_positions['progress_bar'][1])
            cv2.rectangle(frame, (self.text_positions['progress_bar'][0], y_pos), 
                        (self.text_positions['progress_bar'][0] + bar_width, y_pos + 15), 
                        self.colors['progress_bar_bg'], -1)
            cv2.rectangle(frame, (self.text_positions['progress_bar'][0], y_pos), 
                        (self.text_positions['progress_bar'][0] + filled_width, y_pos + 15), 
                        self.colors['progress_bar_fill'], -1)
        
        # Conteo de bostezos
        count = len(self.yawn_times)
        count_color = self.colors['text_critical'] if count >= self.config['MAX_YAWNS_BEFORE_ALERT'] else self.colors['text_normal']
        y_pos = int(h * self.text_positions['yawn_count'][1])
        cv2.putText(frame, f"Bostezos ({self.config['WINDOW_SIZE']//60}min): {count}/{self.config['MAX_YAWNS_BEFORE_ALERT']}", 
                (self.text_positions['yawn_count'][0], y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2)
        
        # Advertencia de múltiples bostezos
        if count >= self.config['MAX_YAWNS_BEFORE_ALERT']:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.colors['overlay_fatigue'], -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            warning_text = f"¡ALERTA! FATIGA DETECTADA - {count} BOSTEZOS EN {self.config['WINDOW_SIZE']//60} MINUTOS"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.putText(frame, warning_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text_critical'], 2)
            cv2.putText(frame, "Se recomienda tomar un descanso", 
                    (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_critical'], 2)
        
        return frame
    
    def reset_yawn_counter(self):
        """Reinicia explícitamente el contador de bostezos y sus banderas asociadas"""
        print("REINICIO MANUAL DEL CONTADOR DE BOSTEZOS")
        self.yawn_times.clear()
        self.third_yawn_alerted = False
        self.report_sent_time = 0
        return True
    
    def get_yawn_count(self):
        """Devuelve el número de bostezos en la ventana temporal"""
        return len(self.yawn_times)