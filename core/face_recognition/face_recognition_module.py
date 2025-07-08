import cv2
import face_recognition
import numpy as np
import pickle
import os
import logging
import time
from core.alarm_module import AlarmModule

# Importar configuración si está disponible
try:
    from config.config_manager import get_config, has_gui
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

class FaceRecognitionModule:
    def __init__(self, operators_dir="operators", config=None):
        """
        Inicializa el módulo de reconocimiento facial (SOLO RECONOCIMIENTO).
        
        Args:
            operators_dir: Directorio de operadores
            config: Configuración personalizada (opcional)
        """
        self.operators_dir = operators_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.operators = {}
        self.logger = logging.getLogger('FaceRecognitionModule')

        # Configuración base (SIMPLIFICADA)
        if config:
            self.config = config
        else:
            # Cargar configuración por defecto
            self.config = {
                'face_tolerance': 0.6,
                'min_confidence': 0.4,
                'recognition_threshold': 0.5,
                'night_mode_threshold': 50,
                'night_tolerance_adjustment': 0.1,
                'enable_night_mode': True,
                'enable_sounds': True,
                'welcome_delay': 5,
                'calibration_confidence': 0.0
            }

        # ========== NUEVO CONTROL DE SESIÓN ==========
        # Control mejorado de sesiones de operadores
        self.operator_sessions = {}  # {operator_id: {'last_seen': timestamp, 'welcomed': bool}}
        self.SESSION_TIMEOUT = 600   # 10 minutos (600 segundos)
        self.MIN_ABSENCE_TIME = 10   # Mínimo 10 segundos ausente para considerar ausencia
        
        # Para operadores no registrados
        self.unknown_last_seen = None
        self.unknown_welcomed = False
        # ============================================
        
        # Estado de iluminación
        self.is_night_mode = False
        self.light_level = 0
        
        # Colores para visualización
        self.colors = {
            'recognized': (0, 255, 0),      # Verde para reconocido
            'unknown': (0, 0, 255),         # Rojo para desconocido
            'text': (255, 255, 255),        # Blanco para texto
            'background': (0, 0, 0)         # Negro para fondo
        }
        
        # Inicializar AlarmModule para audio
        if self.config['enable_sounds']:
            try:
                self.alarm_module = AlarmModule()
                self.alarm_module.initialize()
                self.audio_initialized = True
                self.logger.info("Sistema de audio inicializado a través de AlarmModule")
            except:
                self.logger.warning("No se pudo inicializar el sistema de audio")
                self.audio_initialized = False
                self.alarm_module = None
        else:
            self.audio_initialized = False
            self.alarm_module = None

        self.logger.info(f"FaceRecognitionModule inicializado con control de sesión mejorado")

    def update_config(self, new_config):
        """
        Actualiza la configuración del módulo.
        
        Args:
            new_config: Diccionario con nueva configuración
        """
        self.config.update(new_config)
        self.logger.info("Configuración actualizada")

    def reproducir_audio(self, audio_key):
        """Reproduce un archivo de audio usando AlarmModule"""
        if not self.config['enable_sounds'] or not self.audio_initialized or not self.alarm_module:
            return
            
        try:
            # Mapeo de rutas antiguas a claves del AlarmModule
            audio_map = {
                "assets/audio/bienvenido.mp3": "bienvenido",
                "assets/audio/no_registrado.mp3": "no_registrado"
            }
            
            # Si es una ruta antigua, convertir a clave
            if audio_key in audio_map:
                audio_key = audio_map[audio_key]
            
            # Usar AlarmModule para reproducir
            self.alarm_module.play_audio(audio_key)
            
        except Exception as e:
            self.logger.error(f"Error al reproducir audio: {e}")

    def load_operators(self):
        """Carga operadores desde archivo de encodings"""
        encodings_file = os.path.join(self.operators_dir, "encodings.pkl")

        if not os.path.exists(encodings_file):
            self.logger.warning(f"Archivo de encodings no encontrado: {encodings_file}")
            return False

        try:
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                self.known_face_ids = data['ids']
                self.operators = data['operators']

            self.logger.info(f"Operadores cargados: {len(self.operators)}")
            return True

        except Exception as e:
            self.logger.error(f"Error al cargar operadores: {str(e)}")
            return False

    def identify_operator(self, frame):
        """
        Identifica al operador en el frame actual con control de sesión mejorado.
        
        Returns:
            dict: Información del operador o None si no se reconoce
        """
        if not self.known_face_encodings:
            return None
            
        # Detectar condiciones de iluminación
        if self.config['enable_night_mode']:
            self._detect_lighting_conditions(frame)

        # Ajustar tolerancia según modo día/noche
        current_tolerance = self.config['face_tolerance']
        if self.is_night_mode:
            current_tolerance += self.config['night_tolerance_adjustment']

        # Reducir tamaño para procesamiento más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar rostros
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if not face_locations:
            # No hay rostro detectado - NO reproducir audio
            return None

        try:
            # Obtener encodings de los rostros detectados
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Obtener landmarks faciales
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

            for i, face_encoding in enumerate(face_encodings):
                # Comparar con rostros conocidos
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=current_tolerance
                )

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                # Verificar confianza mínima
                confidence = 1 - face_distances[best_match_index]
                
                # Obtener landmarks si están disponibles
                face_landmarks = face_landmarks_list[i] if i < len(face_landmarks_list) else None
                
                if matches[best_match_index] and confidence >= self.config['min_confidence']:
                    # ========== OPERADOR REGISTRADO ==========
                    operator_id = self.known_face_ids[best_match_index]
                    operator_info = self.operators[operator_id].copy()
                    operator_info['confidence'] = confidence
                    operator_info['is_registered'] = True

                    # Información de ubicación del rostro
                    top, right, bottom, left = face_locations[0]
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    operator_info['face_location'] = (top, right, bottom, left)
                    operator_info['face_area'] = (right - left) * (bottom - top)
                    
                    # Agregar landmarks
                    if face_landmarks:
                        scaled_landmarks = {}
                        for feature, points in face_landmarks.items():
                            scaled_landmarks[feature] = [(p[0] * 4, p[1] * 4) for p in points]
                        operator_info['face_landmarks'] = scaled_landmarks

                    # ========== CONTROL DE SESIÓN MEJORADO ==========
                    current_time = time.time()
                    
                    # Verificar si tenemos historial de este operador
                    if operator_id in self.operator_sessions:
                        session_info = self.operator_sessions[operator_id]
                        time_since_last_seen = current_time - session_info['last_seen']
                        
                        # Determinar si debemos dar bienvenida
                        if time_since_last_seen > self.SESSION_TIMEOUT:
                            # Ausente más de 10 minutos - Nueva sesión
                            self.reproducir_audio("assets/audio/bienvenido.mp3")
                            self.logger.info(f"Nueva sesión para {operator_info['name']} después de {time_since_last_seen/60:.1f} minutos")
                            session_info['welcomed'] = True
                        elif time_since_last_seen >= self.MIN_ABSENCE_TIME and not session_info['welcomed']:
                            # Caso especial: si nunca fue bienvenido en esta sesión
                            self.reproducir_audio("assets/audio/bienvenido.mp3")
                            session_info['welcomed'] = True
                        # Si ausente menos de 10 segundos o ya fue bienvenido - no hacer nada
                        
                        session_info['last_seen'] = current_time
                    else:
                        # Primera vez que vemos a este operador
                        self.reproducir_audio("assets/audio/bienvenido.mp3")
                        self.logger.info(f"Primera detección de {operator_info['name']}")
                        self.operator_sessions[operator_id] = {
                            'last_seen': current_time,
                            'welcomed': True
                        }
                    
                    # Reset el estado de operador desconocido
                    self.unknown_last_seen = None
                    self.unknown_welcomed = False

                    return operator_info
                    
                else:
                    # ========== OPERADOR NO REGISTRADO ==========
                    top, right, bottom, left = face_locations[0]
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    unknown_info = {
                        'id': 'UNKNOWN',
                        'name': 'No Registrado',
                        'confidence': confidence,
                        'is_registered': False,
                        'face_location': (top, right, bottom, left),
                        'face_area': (right - left) * (bottom - top),
                        'best_match_distance': float(face_distances[best_match_index])
                    }
                    
                    # Agregar landmarks
                    if face_landmarks:
                        scaled_landmarks = {}
                        for feature, points in face_landmarks.items():
                            scaled_landmarks[feature] = [(p[0] * 4, p[1] * 4) for p in points]
                        unknown_info['face_landmarks'] = scaled_landmarks
                    
                    # ========== CONTROL DE SESIÓN PARA NO REGISTRADO ==========
                    current_time = time.time()
                    
                    if self.unknown_last_seen is not None:
                        time_since_last_seen = current_time - self.unknown_last_seen
                        
                        if time_since_last_seen > self.SESSION_TIMEOUT:
                            # Ausente más de 10 minutos
                            self.reproducir_audio("assets/audio/no_registrado.mp3")
                            self.logger.warning(f"Operador no registrado detectado nuevamente después de {time_since_last_seen/60:.1f} minutos")
                            self.unknown_welcomed = True
                        elif time_since_last_seen >= self.MIN_ABSENCE_TIME and not self.unknown_welcomed:
                            # Primera vez en esta sesión
                            self.reproducir_audio("assets/audio/no_registrado.mp3")
                            self.unknown_welcomed = True
                        # Si menos de 10 segundos o ya avisado - no hacer nada
                    else:
                        # Primera vez detectando operador no registrado
                        self.reproducir_audio("assets/audio/no_registrado.mp3")
                        self.logger.warning("Operador no registrado detectado por primera vez")
                        self.unknown_welcomed = True
                    
                    self.unknown_last_seen = current_time
                    
                    return unknown_info

        except Exception as e:
            self.logger.error(f"Error en reconocimiento facial: {str(e)}")

        return None
        
    def _detect_lighting_conditions(self, frame):
        """Detecta las condiciones de iluminación para ajustar tolerancias"""
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
    
    def draw_operator_info(self, frame, operator_info):
        """
        Dibuja los puntos faciales (landmarks) usados para reconocimiento.
        NO dibuja ojos ni boca (reservados para otros módulos).
        
        Args:
            frame: Frame donde dibujar
            operator_info: Información del operador
            
        Returns:
            frame: Frame con información dibujada
        """
        if operator_info and 'face_location' in operator_info:
            # Colores según si está registrado o no
            if operator_info.get('is_registered', False):
                points_color = (0, 255, 0)       # Verde para puntos
                lines_color = (0, 255, 0)        # Verde para líneas
                text_color = self.colors['recognized']
            else:
                points_color = (0, 0, 255)       # Rojo para puntos
                lines_color = (0, 0, 255)        # Rojo para líneas
                text_color = self.colors['unknown']
            
            # Dibujar landmarks faciales si están disponibles
            if 'face_landmarks' in operator_info:
                landmarks = operator_info['face_landmarks']
                
                # Definir conexiones entre puntos para crear la malla facial
                # EXCLUYENDO OJOS Y BOCA
                
                # 1. Contorno de la mandíbula
                if 'chin' in landmarks:
                    chin_points = landmarks['chin']
                    for i in range(len(chin_points) - 1):
                        cv2.line(frame, chin_points[i], chin_points[i+1], lines_color, 1)
                
                # 2. Ceja izquierda
                if 'left_eyebrow' in landmarks:
                    eyebrow_points = landmarks['left_eyebrow']
                    for i in range(len(eyebrow_points) - 1):
                        cv2.line(frame, eyebrow_points[i], eyebrow_points[i+1], lines_color, 1)
                
                # 3. Ceja derecha
                if 'right_eyebrow' in landmarks:
                    eyebrow_points = landmarks['right_eyebrow']
                    for i in range(len(eyebrow_points) - 1):
                        cv2.line(frame, eyebrow_points[i], eyebrow_points[i+1], lines_color, 1)
                
                # 4. Puente de la nariz
                if 'nose_bridge' in landmarks:
                    nose_points = landmarks['nose_bridge']
                    for i in range(len(nose_points) - 1):
                        cv2.line(frame, nose_points[i], nose_points[i+1], lines_color, 1)
                
                # 5. Parte inferior de la nariz
                if 'nose_tip' in landmarks:
                    nose_tip = landmarks['nose_tip']
                    for i in range(len(nose_tip) - 1):
                        cv2.line(frame, nose_tip[i], nose_tip[i+1], lines_color, 1)
                
                # NO DIBUJAR: left_eye, right_eye, top_lip, bottom_lip
                
                # Dibujar solo los puntos de las características permitidas
                allowed_features = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip']
                
                for feature in allowed_features:
                    if feature in landmarks:
                        points = landmarks[feature]
                        for (x, y) in points:
                            # Punto principal
                            cv2.circle(frame, (x, y), 2, points_color, -1)
                            # Borde del punto para mejor visibilidad
                            cv2.circle(frame, (x, y), 3, (255, 255, 255), 1)
            
            # Información del operador (abajo del rostro)
            top, right, bottom, left = operator_info['face_location']
            name = operator_info.get('name', 'Desconocido')
            confidence = operator_info.get('confidence', 0)
            
            # Fondo para el texto
            text = f"{name} ({confidence:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # Rectángulo de fondo
            cv2.rectangle(frame, 
                         (left, bottom + 5), 
                         (left + text_size[0] + 10, bottom + 30),
                         self.colors['background'], -1)
            
            # Texto
            cv2.putText(frame, text, (left + 5, bottom + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            # Mostrar modo (día/noche) en esquina superior
            mode_text = "MODO: NOCHE" if self.is_night_mode else "MODO: DIA"
            cv2.putText(frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
        return frame
    
    def get_status(self):
        """Obtiene el estado actual del módulo"""
        return {
            'operators_loaded': len(self.operators),
            'is_night_mode': self.is_night_mode,
            'light_level': self.light_level,
            'calibration_confidence': self.config.get('calibration_confidence', 0),
            'current_tolerance': self.config['face_tolerance'] + (
                self.config['night_tolerance_adjustment'] if self.is_night_mode else 0
            ),
            'active_sessions': len(self.operator_sessions),
            'session_timeout_minutes': self.SESSION_TIMEOUT / 60
        }