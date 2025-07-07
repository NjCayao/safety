import os
import time
from collections import deque
import pygame
import cv2
import dlib
from scipy.spatial import distance
import numpy as np

# 🆕 NUEVO: Importar sistema de configuración
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Sistema de configuración no disponible, usando valores por defecto")

class FatigueDetector:
    def __init__(self, model_path, headless=False):
        """Inicializa el detector de fatiga con los archivos de audio disponibles"""
        self.headless = headless
        
        # 🆕 NUEVO: Cargar configuración externa (con fallbacks seguros)
        if CONFIG_AVAILABLE:
            # Configuración de tiempos (en segundos)
            self.EYE_CLOSED_THRESHOLD = get_config('fatigue.eye_closed_threshold', 1.5)
            self.WINDOW_SIZE = get_config('fatigue.window_size', 600)
            self.ALARM_COOLDOWN = get_config('fatigue.alarm_cooldown', 5)
            
            # Umbral EAR (Eye Aspect Ratio)
            self.EAR_THRESHOLD = get_config('fatigue.ear_threshold', 0.20)
            self.EAR_NIGHT_ADJUSTMENT = get_config('fatigue.ear_night_adjustment', 0.03)
            
            # Configuración de modo nocturno
            self.night_mode_threshold = get_config('fatigue.night_mode_threshold', 50)
            self.enable_night_mode = get_config('fatigue.enable_night_mode', True)
            
            # Configuración de suavizado
            self.frames_to_confirm = get_config('fatigue.frames_to_confirm', 2)
            self.calibration_period = get_config('fatigue.calibration_period', 30)
            
            # Configuración de GUI
            self.show_gui = has_gui()
            
            print(f"✅ Configuración cargada - Umbral EAR: {self.EAR_THRESHOLD}, GUI: {self.show_gui}")
        else:
            # ✅ FALLBACK: Valores originales si no hay configuración
            self.EYE_CLOSED_THRESHOLD = 1.5
            self.WINDOW_SIZE = 600
            self.ALARM_COOLDOWN = 5
            self.EAR_THRESHOLD = 0.25
            self.EAR_NIGHT_ADJUSTMENT = 0.03
            self.night_mode_threshold = 50
            self.enable_night_mode = True
            self.frames_to_confirm = 2
            self.calibration_period = 30
            self.show_gui = True  # Default para compatibilidad
            
            print("⚠️ Usando configuración por defecto (hardcodeada)")
        
        # ✅ RESTO DEL CÓDIGO ORIGINAL INTACTO
        # Estado del detector
        self.eyes_closed_duration = 0.0
        self.microsleeps = deque()  # Cola para manejar la ventana temporal
        self.last_alarm_time = 0
        self.microsleep_in_progress = False  # Flag para controlar detección continua
        
        # Tiempo para medir duración real de ojos cerrados
        self.eyes_closed_start_time = None
        
        # Para suavizar la detección
        self.closed_frames = 0
        self.open_frames = 0
        self.last_ear_values = deque(maxlen=3)  # Mantener últimos valores EAR

        # Conteo de parpadeos
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.last_blink_time = 0
        self.blink_duration_threshold = 0.5  # Máximo 500ms para ser parpadeo

        # Valores mínimos y máximos de EAR observados (para calibración)
        self.min_ear_observed = 1.0
        self.max_ear_observed = 0.0
        self.calibration_frame_count = 0
        
        # Detección de condiciones de iluminación - NUEVO
        self.is_night_mode = False
        self.light_level = 0
        
        # Configuración de modelos
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(model_path)
        
        # Inicializar sistema de audio
        self._initialize_audio_system()
        
        # Mensajes en pantalla
        self.display_messages = []
        self.DISPLAY_TIME = 3.0  # Tiempo en segundos para mostrar mensajes
        
        # Estado de debugging
        self.last_status_time = 0
        print("Detector de fatiga inicializado. UMBRAL EAR:", self.EAR_THRESHOLD)

    def _initialize_audio_system(self):
        """Configura el sistema de audio con manejo de errores"""
        try:
            pygame.mixer.init()
            
            # Alarma general
            self.alarm_sound = self._load_audio_file("alarma.mp3")
            
            # Sistema de mensajes de voz adaptado para cada nivel
            self.voice_messages = {
                1: self._load_audio_file("fatigue_1.mp3"),  # Primera advertencia
                2: self._load_audio_file("fatigue_2.mp3"),  # Segunda advertencia
                3: self._load_audio_file("fatigue_3.mp3")   # Tercera advertencia (crítica)
            }
            
            # Fallback a recomendacion_pausas.mp3 si no se encuentran los específicos
            for level in [1, 2, 3]:
                if not self.voice_messages[level]:
                    self.voice_messages[level] = self._load_audio_file("recomendacion_pausas.mp3")
                    print(f"Usando sonido alternativo para nivel {level}")
            
            print("Sistema de audio inicializado correctamente")
        except Exception as e:
            print(f"Error inicializando sistema de audio: {str(e)}")
            self.alarm_sound = None
            self.voice_messages = {}
    
    
    def _load_audio_file(self, filename):
        """Carga un archivo de audio con manejo de errores"""
        try:
            # Obtener la ruta absoluta desde la raíz del proyecto
            import os
            import sys
            
            # Subir dos niveles desde core/fatigue/ hasta la raíz
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            # Construir ruta al archivo de audio
            path = os.path.join(project_root, "assets", "audio", filename)
            
            if os.path.exists(path):
                print(f"Archivo de audio cargado: {filename}")
                return pygame.mixer.Sound(path)
            print(f"Advertencia: Archivo de audio no encontrado - {path}")
            return None
        except Exception as e:
            print(f"Error cargando {filename}: {str(e)}")
            return None
    
    def _calculate_head_pose(self, landmarks):
        """Calcula la orientación de la cabeza para evitar falsos positivos"""
        # Puntos clave para determinar orientación
        nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
        chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
        
        # Vector vertical de la cara
        face_vertical = chin - nose_tip
        face_angle = np.arctan2(face_vertical[1], face_vertical[0])
        
        # Ángulo respecto a vertical (90 grados)
        vertical_deviation = abs(face_angle - np.pi/2)
        
        # Si la cabeza está muy inclinada, es probablemente una maniobra
        is_looking_down = vertical_deviation > 0.3  # ~17 grados
        
        return is_looking_down, vertical_deviation
    
    def detect(self, frame):
        """Versión mejorada del método detect"""
        current_time = time.time()
        
        # Conversión a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar nivel de iluminación
        if self.enable_night_mode:
            self._detect_lighting_conditions(gray)
        
        # Mejora de imagen
        enhanced_gray = self._enhance_image(gray)
        
        # Detección facial
        faces = self.face_detector(enhanced_gray, 0)
        
        microsleep_detected = False
        critical_fatigue = False
        
        if self.show_gui:
            frame = self._draw_mode_indicator(frame)
        
        if not faces:
            # Sin rostro
            if self.eyes_closed_start_time is not None:
                self.eyes_closed_start_time = None
                self.eyes_closed_duration = 0
            
            if self.show_gui:
                frame = self._draw_no_face_info(frame)
            else:
                if current_time - self.last_status_time > 5:
                    print("⚠️ No se detecta rostro")
                    self.last_status_time = current_time
            
            return False, False, frame
        
        landmarks = self.landmark_predictor(enhanced_gray, faces[0])
        
        # NUEVA: Detectar orientación de cabeza
        is_looking_down, head_angle = self._calculate_head_pose(landmarks)
        
        # Detección de ojos
        left_eye = self._get_eye_points(landmarks, 36, 42)
        right_eye = self._get_eye_points(landmarks, 42, 48)
        
        # Calcular EAR
        ear_left = self._calculate_ear(left_eye)
        ear_right = self._calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0
        
        # Calibración automática
        if self.calibration_frame_count < self.calibration_period:
            self.calibration_frame_count += 1
            self.min_ear_observed = min(self.min_ear_observed, ear)
            self.max_ear_observed = max(self.max_ear_observed, ear)
            
            if self.calibration_frame_count == self.calibration_period:
                range_ear = self.max_ear_observed - self.min_ear_observed
                if range_ear > 0.1:
                    new_threshold = self.min_ear_observed + (range_ear * 0.3)
                    print(f"Calibración completada. Nuevo umbral EAR: {new_threshold:.2f}")
                    # self.EAR_THRESHOLD = new_threshold
        
        # Añadir a valores recientes
        self.last_ear_values.append(ear)
        avg_ear = sum(self.last_ear_values) / len(self.last_ear_values)
        
        # Obtener umbral actual
        current_threshold = self._get_current_ear_threshold()
        
        # IMPORTANTE: Ajustar umbral si está mirando hacia abajo
        if is_looking_down:
            # Reducir umbral cuando mira hacia abajo (hacer menos sensible)
            adjusted_threshold = current_threshold * 0.7
            print(f"Cabeza inclinada detectada - Umbral ajustado: {adjusted_threshold:.2f}")
        else:
            adjusted_threshold = current_threshold
        
        # Determinar si los ojos están abiertos/cerrados
        eyes_open = avg_ear > adjusted_threshold
        
        # Confirmación por múltiples frames
        if eyes_open:
            self.open_frames += 1
            self.closed_frames = 0
        else:
            # Solo contar como cerrado si NO está mirando hacia abajo
            # O si está mirando hacia abajo Y el EAR es MUY bajo
            if not is_looking_down or ear < adjusted_threshold * 0.8:
                self.closed_frames += 1
                self.open_frames = 0
            else:
                # Probablemente solo está mirando hacia abajo
                self.open_frames += 1
                self.closed_frames = 0
        
        # Confirmación
        if self.closed_frames >= self.frames_to_confirm:
            if self._is_looking_down(landmarks):
                confirmed_eyes_closed = False
                self.closed_frames = 0
                print("Resetando contador - solo está mirando hacia abajo")
            else:
                confirmed_eyes_closed = True
        else:
            confirmed_eyes_closed = False
        confirmed_eyes_open = self.open_frames >= self.frames_to_confirm
        
        # MÉTODO DE CRONOMETRAJE PRECISO: Usar timestamps reales para calcular duración
        if confirmed_eyes_closed:
            # Iniciar contador si es la primera vez que se detectan ojos cerrados
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
                print(f"Ojos cerrados detectados - iniciando contador")
            
            # Calcular duración real con timestamps
            self.eyes_closed_duration = current_time - self.eyes_closed_start_time
            
            # Imprimir estado periódicamente
            if int(self.eyes_closed_duration * 10) % 5 == 0:  # Cada 0.5 segundos
                mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                print(f"⚠️ Ojos cerrados por {self.eyes_closed_duration:.1f} segundos (EAR: {ear:.2f}, Umbral: {current_threshold:.2f}, Modo: {mode_str})")
                
            # PUNTO CRÍTICO: Verificar si alcanzamos EXACTAMENTE el umbral de microsueño
            # y no estamos ya en un microsueño en progreso
            if self.eyes_closed_duration >= self.EYE_CLOSED_THRESHOLD and not self.microsleep_in_progress:
                # Detectar microsueño
                print(f"⚠️⚠️⚠️ MICROSUEÑO DETECTADO: Ojos cerrados por {self.eyes_closed_duration:.1f} segundos")
                microsleep_detected = True
                self.microsleep_in_progress = True
                
                # Registrar microsueño
                self._register_microsleep(current_time)
                
                # Activar alarmas inmediatamente (sin verificar cooldown para el primer frame)
                self._trigger_alarms(current_time)
                
                # 🆕 NUEVO: Solo agregar mensaje visual si GUI está habilitada
                if self.show_gui:
                    self._add_display_message(f"¡MICROSUEÑO DETECTADO! ({len(self.microsleeps)}/3)", (0, 0, 255))
                
        elif confirmed_eyes_open:
            # Si los ojos estaban cerrados, reportar duración
            if self.eyes_closed_start_time is not None:
                final_duration = current_time - self.eyes_closed_start_time
                print(f"Ojos abiertos después de {final_duration:.1f} segundos")

                # NUEVO: Detectar si fue un parpadeo
                if final_duration < self.blink_duration_threshold:
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    print(f"Parpadeo detectado #{self.blink_count}")
                
                # Resetear contador y tiempo de inicio
                self.eyes_closed_duration = 0
                self.eyes_closed_start_time = None
                
                # Resetear flag de microsueño en progreso
                if self.microsleep_in_progress:
                    self.microsleep_in_progress = False
        
        # Verificar si tenemos 3 o más microsueños (fatiga crítica)
        critical_fatigue = len(self.microsleeps) >= 3
        
        # SIEMPRE dibujar información de los ojos
        frame = self._draw_eye_info(frame, left_eye, right_eye, ear, avg_ear, current_threshold, current_time)
        frame = self._draw_display_messages(frame, current_time)

        # Información de orientación de cabeza
        orientation_text = "Mirando abajo" if is_looking_down else "Mirando al frente"
        cv2.putText(frame, orientation_text, (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Solo hacer log en consola si es headless
        if not self.show_gui:
            # En modo headless, log periódico del estado
            if current_time - self.last_status_time > 10:
                status = "OJOS CERRADOS" if not eyes_open else "OJOS ABIERTOS"
                mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                print(f"📊 Estado: {status} | EAR: {ear:.2f} | Umbral: {current_threshold:.2f} | Modo: {mode_str} | Microsueños: {len(self.microsleeps)}/3")
                self.last_status_time = current_time
        
        return microsleep_detected, critical_fatigue, frame
    
    # NUEVO: Métodos para gestionar el modo nocturno
    def _detect_lighting_conditions(self, gray_frame):
        """Detecta las condiciones de iluminación y determina si es modo nocturno"""
        # Calcular nivel promedio de iluminación (0-255)
        self.light_level = np.mean(gray_frame)
        
        # Determinar si estamos en modo nocturno
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.night_mode_threshold
        
        # Notificar cambio de modo
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
            print(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
    
    def _enhance_image(self, gray_frame):
        """Mejora la imagen según condiciones de iluminación"""
        # En modo nocturno, aplicar más mejoras para infrarrojo
        if self.is_night_mode:
            # Normalizar histograma para mejorar contraste en IR
            enhanced = cv2.equalizeHist(gray_frame)
            
            # Reducir ruido para imágenes IR
            enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        else:
            # En modo diurno, mejora de contraste más suave
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray_frame)
        
        return enhanced
    
    def _get_current_ear_threshold(self):
        """Obtiene el umbral EAR ajustado según el modo (día/noche)"""
        if self.is_night_mode:
            # En modo nocturno, hacemos el umbral más permisivo
            return self.EAR_THRESHOLD - self.EAR_NIGHT_ADJUSTMENT
        else:
            return self.EAR_THRESHOLD
    
    def _draw_mode_indicator(self, frame):
        """Dibuja indicador de modo (día/noche)"""
        h, w = frame.shape[:2]
        mode_str = "MODO NOCTURNO" if self.is_night_mode else "MODO DIURNO"
        mode_color = (0, 150, 255) if self.is_night_mode else (255, 200, 0)
        
        # Fondo para el indicador
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-40), (200, h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, mode_str, (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        return frame

    def _register_microsleep(self, timestamp):
        """Registra un nuevo microsueño y limpia los antiguos"""
        self.microsleeps.append(timestamp)
        
        # Eliminar eventos fuera de la ventana de 10 minutos
        while self.microsleeps and (timestamp - self.microsleeps[0] > self.WINDOW_SIZE):
            self.microsleeps.popleft()

        # Realizar advertencia si alcanzamos 3 microsueños
        if len(self.microsleeps) == 3:
            print("¡ALERTA! Se alcanzaron 3 microsueños en los últimos 10 minutos.")
            # Generar advertencia crítica
            self._trigger_alarms(timestamp)

        # Resetear el contador si alcanzamos o superamos 3 microsueños
        if len(self.microsleeps) >= 3:
            print("¡CONTADOR RESETEADO! Se alcanzaron 3 microsueños.")
            # Mantener solo el microsueño más reciente
            most_recent = self.microsleeps[-1]
            self.microsleeps.clear()
            self.microsleeps.append(most_recent)
            print("Nuevo conteo de microsueños: 1/3")

        print(f"¡MICROSUEÑO REGISTRADO! Total en los últimos 10 minutos: {len(self.microsleeps)}")
   
    def reproducir_mensaje_voz(self, path):
        try:
            if os.path.exists(path):
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                print(f"Reproduciendo mensaje de voz: {os.path.basename(path)}")
                while pygame.mixer.music.get_busy():  # Esperar hasta que termine
                    pygame.time.Clock().tick(10)
            else:
                print(f"⚠️ Archivo no encontrado: {path}")
        except Exception as e:
            print(f"❌ Error al reproducir mensaje de voz: {e}")

    def _trigger_alarms(self, current_time):
        """Activa la alarma y los mensajes de fatiga según el conteo"""
        self.last_alarm_time = current_time
        microsleep_count = len(self.microsleeps)
        
        print(f"¡ALERTA! Microsueño detectado #{microsleep_count} en los últimos 10 minutos")
        
        # Obtener ruta correcta para los audios
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Reproducir la alarma
        alarm_audio = os.path.join(project_root, "assets", "audio", "alarma.mp3")
        
        if self.alarm_sound and os.path.exists(alarm_audio):
            try:
                pygame.mixer.stop()
                self.alarm_sound = pygame.mixer.Sound(alarm_audio)
                self.alarm_sound.play()
                print(f"Reproduciendo alarma: {alarm_audio}")
                time.sleep(1)
            except Exception as e:
                print(f"Error al reproducir alarma: {str(e)}")
        else:
            print(f"Error: No se encontró el archivo de alarma {alarm_audio}")

        # Reproducir el mensaje de fatiga correspondiente
        if microsleep_count == 1:
            fatigue_file = "fatigue_1.mp3"
        elif microsleep_count == 2:
            fatigue_file = "fatigue_2.mp3"
        elif microsleep_count >= 3:
            fatigue_file = "fatigue_3.mp3"
        
        fatigue_audio = os.path.join(project_root, "assets", "audio", fatigue_file)
        
        if os.path.exists(fatigue_audio):
            try:
                self.reproducir_mensaje_voz(fatigue_audio)
                print(f"Reproduciendo mensaje de fatiga: {fatigue_audio}")
            except Exception as e:
                print(f"Error al reproducir mensaje de fatiga: {str(e)}")

    def _send_critical_report(self):
        """Envía reporte al servidor (implementar conexión real)"""
        print("Enviando reporte crítico al servidor...")
        # Aquí iría el código para enviar el reporte
        # Esta función será llamada por el sistema principal

    def _get_eye_points(self, landmarks, start, end):
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]

    def _calculate_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def _is_looking_down(self, landmarks):
        """Detecta si la persona está mirando hacia abajo"""
        try:
            # Puntos clave
            nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
            nose_bridge = np.array([landmarks.part(27).x, landmarks.part(27).y])
            left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
            right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
            
            # Centro de los ojos
            eyes_center = (left_eye + right_eye) / 2
            
            # Si la punta de la nariz está más abajo que el centro de los ojos + margen
            # probablemente está mirando hacia abajo
            vertical_diff = nose_tip[1] - eyes_center[1]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Si la nariz está más de 40% de la distancia entre ojos por debajo
            is_looking_down = vertical_diff > (eye_distance * 0.4)
            
            if is_looking_down:
                print("👀 Mirando hacia abajo detectado - NO es microsueño")
                
            return is_looking_down
        except:
            return False
    
    def _draw_no_face_info(self, frame):
        """Dibuja información cuando no se detecta rostro"""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente para el mensaje principal
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "NO SE DETECTA ROSTRO", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Mostrar información del modo actual
        mode_str = "MODO NOCTURNO (IR)" if self.is_night_mode else "MODO DIURNO"
        cv2.putText(frame, f"Detección en {mode_str}", (10, h-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        
        # Instrucciones para el usuario
        cv2.putText(frame, "Ajuste su posición frente a la cámara", (10, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Mantener visualización del conteo de microsueños
        count_color = (0, 0, 255) if len(self.microsleeps) >= 2 else (0, 255, 0)
        cv2.putText(frame, f"Microsueños (10min): {len(self.microsleeps)}/3", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)
        
        return frame

    def _draw_eye_info(self, frame, left_eye, right_eye, ear, avg_ear, current_threshold, current_time):
        """Dibuja información visual sobre los ojos"""
        # Dibujar contornos de ojos
        cv2.drawContours(frame, [cv2.convexHull(np.array(left_eye))], -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(np.array(right_eye))], -1, (0,255,0), 1)

        # Dibujar puntos individuales del ojo
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Puntos rojos
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Puntos rojos
        
        # Determinar estado de los ojos usando el umbral ajustado al modo actual
        status = "OJOS ABIERTOS" if avg_ear > current_threshold else "OJOS CERRADOS"
        color = (0, 255, 0) if avg_ear > current_threshold else (0, 0, 255)
        
        # Mostrar EAR y estado
        mode_str = "noche" if self.is_night_mode else "día"
        cv2.putText(frame, f"EAR: {ear:.2f} (Umbral: {current_threshold:.2f}, Modo: {mode_str})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Estado: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Indicador de tiempo con ojos cerrados
        if self.eyes_closed_duration > 0:
            # Cambiar color a amarillo cuando está cerca del umbral
            time_color = (0, 165, 255) if self.eyes_closed_duration > (self.EYE_CLOSED_THRESHOLD * 0.7) else (255, 255, 255)
            
            # Mostrar progreso hacia umbral de microsueño
            progress = min(self.eyes_closed_duration / self.EYE_CLOSED_THRESHOLD, 1.0) * 100
            cv2.putText(frame, f"Tiempo ojos cerrados: {self.eyes_closed_duration:.1f}s / {self.EYE_CLOSED_THRESHOLD}s ({progress:.0f}%)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
            
            # Barra de progreso visual para tiempo con ojos cerrados
            bar_width = 200
            filled_width = int(bar_width * (self.eyes_closed_duration / self.EYE_CLOSED_THRESHOLD))
            cv2.rectangle(frame, (10, 100), (10 + bar_width, 115), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 100), (10 + filled_width, 115), time_color, -1)
        
        # Mostrar conteo de microsueños con advertencia visual
        count_color = (0, 0, 255) if len(self.microsleeps) >= 2 else (255, 255, 255)
        y_position = 140  # Posición fija para el conteo
        
        if len(self.microsleeps) >= 3:
            # Alerta visual más prominente para fatiga crítica
            cv2.putText(frame, f"¡ALERTA CRÍTICA! {len(self.microsleeps)} MICROSUEÑOS/10min", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Añadir overlay rojo transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Mensaje de advertencia centrado
            text = "¡FATIGA CRÍTICA DETECTADA!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
        else:
            # Visualización normal del conteo
            cv2.putText(frame, f"Microsueños (10min): {len(self.microsleeps)}/3", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)
        
        # Calcular y mostrar parpadeos por minuto
        elapsed_time = current_time - self.blink_start_time
        if elapsed_time > 0:
            blinks_per_minute = (self.blink_count / elapsed_time) * 60
            cv2.putText(frame, f"Parpadeos/min: {int(blinks_per_minute)}", 
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _add_display_message(self, message, color=(255, 255, 255)):
        """Añade un mensaje temporal para mostrar en pantalla"""
        self.display_messages.append({
            'text': message,
            'color': color,
            'time': time.time(),
            'duration': self.DISPLAY_TIME
        })
    
    def _draw_display_messages(self, frame, current_time):
        """Dibuja mensajes temporales en pantalla"""
        # Filtrar mensajes expirados
        self.display_messages = [m for m in self.display_messages 
                                if current_time - m['time'] < m['duration']]
        
        # Dibujar mensajes activos en la parte superior de la pantalla
        y_offset = 250  # Posición inicial
        for msg in self.display_messages:
            # Calcular transparencia basada en tiempo restante
            remaining = 1.0 - (current_time - msg['time']) / msg['duration']
            alpha = min(1.0, remaining * 1.5)  # Fade out effect
            
            # Ajustar color con transparencia
            color = tuple(int(c * alpha) for c in msg['color'])
            
            # Dibujar fondo para mensaje (mejora visual)
            text_size = cv2.getTextSize(msg['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            # Fondo semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                          (text_x - 10, y_offset - 25), 
                          (text_x + text_size[0] + 10, y_offset + 5), 
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Dibujar mensaje centrado
            cv2.putText(frame, msg['text'], (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
        
        return frame
        
    def get_microsleep_count(self):
        """Devuelve el número actual de microsueños en la ventana de tiempo"""
        return len(self.microsleeps)
        
    # Método para forzar un microsueño (solo para pruebas)
    def force_microsleep(self):
        """Fuerza un microsueño para pruebas"""
        current_time = time.time()
        self._register_microsleep(current_time)
        self._trigger_alarms(current_time)
        print("MICROSUEÑO FORZADO PARA PRUEBAS")
        return True, len(self.microsleeps) >= 3
    
    def update_thresholds(self, new_thresholds):
        """
        Actualiza los umbrales del detector con valores personalizados.
        
        Args:
            new_thresholds: Diccionario con nuevos umbrales
        """
        if 'ear_threshold' in new_thresholds:
            self.EAR_THRESHOLD = new_thresholds['ear_threshold']
            print(f"Umbral EAR actualizado a: {self.EAR_THRESHOLD}")
        
        if 'ear_night_adjustment' in new_thresholds:
            self.EAR_NIGHT_ADJUSTMENT = new_thresholds['ear_night_adjustment']
        
        if 'microsleep_threshold' in new_thresholds:
            self.EYE_CLOSED_THRESHOLD = new_thresholds['microsleep_threshold']
        
        # Actualizar cualquier otro umbral que uses
        print(f"Umbrales actualizados: EAR={self.EAR_THRESHOLD}, Microsueño={self.EYE_CLOSED_THRESHOLD}s")

    def get_statistics(self):
        elapsed_time = time.time() - self.blink_start_time
        blinks_per_minute = (self.blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        """
        Obtiene estadísticas del detector.
        
        Returns:
            dict: Estadísticas actuales
        """
        return {
            'total_microsleeps': len(self.microsleeps),
            'total_blinks': self.blink_count,  # Si no trackeas blinks, dejar en 0
            'average_blink_rate': blinks_per_minute,
            'analysis_duration': elapsed_time,
            'recent_microsleeps': len(self.microsleeps)
        }
    
    def reset(self):
        """Reinicia el detector"""
        self.eyes_closed_duration = 0.0
        self.microsleeps.clear()
        self.last_alarm_time = 0
        self.microsleep_in_progress = False
        self.eyes_closed_start_time = None
        self.closed_frames = 0
        self.open_frames = 0
        self.last_ear_values.clear()
        self.min_ear_observed = 1.0
        self.max_ear_observed = 0.0
        self.calibration_frame_count = 0
        print("Detector de fatiga reiniciado")