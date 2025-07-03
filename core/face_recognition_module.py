import cv2
import face_recognition
import numpy as np
import pickle
import os
import logging
import pygame
import time
from collections import deque
from scipy.spatial import distance as dist

class FaceRecognitionModule:
    def __init__(self, operators_dir="operators", tolerance=0.6):
        self.operators_dir = operators_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.operators = {}
        self.logger = logging.getLogger('FaceRecognitionModule')


        # AGREGAR ESTO PARA DEBUG:
        print(f"🔍 FaceRecognitionModule inicializado con:")
        print(f"   - operators_dir: {self.operators_dir}")
        print(f"   - Ruta absoluta: {os.path.abspath(self.operators_dir)}")
        print(f"   - ¿Existe?: {os.path.exists(self.operators_dir)}")

        # Inicializar pygame para audio
        pygame.mixer.init()
        self.ultimo_operador_id = None  # Para no repetir audio innecesariamente
        
        # Configuración para modo diurno/nocturno
        self.is_night_mode = False
        self.light_level = 0
        self.night_mode_threshold = 50  # Umbral para modo nocturno (0-255)
        
        # Deshabilitar sonidos si es necesario
        self.enable_sounds = False  # Cambiar a True para habilitar los sonidos
        
        # Personalización de colores según modo
        self.day_colors = {
            'name': (255, 0, 0),         # Rojo para nombres
            'landmarks': (0, 255, 255),  # Amarillo para puntos faciales
            'expression': (255, 0, 0),   # Azul para textos de expresiones
            'background': (0, 0, 0, 0.5) # Fondo negro semitransparente
        }
        
        self.night_colors = {
            'name': (255, 0, 0),         # Rojo para nombres en modo noche
            'landmarks': (0, 180, 180),  # Amarillo suave para puntos faciales
            'expression': (255, 0, 0),   # Azul para textos de expresiones
            'background': (0, 0, 0, 0.5) # Fondo negro semitransparente
        }
        
        # Variables para análisis de expresiones faciales
        self.expresion_enabled = True
        self.expresion_sensitivity = 0.7
        self.expresion_memory_frames = 10
        self.expresion_threshold = 0.65
        self.expresion_history = deque(maxlen=10)
        self.current_expresion = "neutral"
        
        # Variables para detección de fatiga
        self.fatiga_enabled = True
        self.fatiga_ear_threshold = 0.25  # Eye Aspect Ratio threshold
        self.fatiga_time_window = 60  # segundos
        self.blink_counter = 0
        self.blink_history = []
        self.last_blink_time = time.time()
        self.ear_history = deque(maxlen=30)
        self.fatiga_percentage = 0
        
        # Variables para detección de estrés
        self.estres_enabled = True
        self.estres_facial_tension_threshold = 0.6
        self.estres_micro_movement_threshold = 0.3
        self.estres_time_window = 30
        self.facial_movement_history = deque(maxlen=20)
        self.last_landmarks = None
        self.estres_percentage = 0
        
        # No reproducir alarma.mp3 al inicio
        self.audio_initialized = False

    def reproducir_audio(self, ruta):
        """Reproduce un archivo de audio si los sonidos están habilitados"""
        if not self.enable_sounds:
            return
            
        try:
            if os.path.exists(ruta):
                pygame.mixer.music.load(ruta)
                pygame.mixer.music.play()
            else:
                self.logger.warning(f"Archivo de audio no encontrado: {ruta}")
        except Exception as e:
            self.logger.error(f"Error al reproducir audio: {e}")

    def load_operators(self):
        """Carga operadores desde archivo de encodings"""
        encodings_file = os.path.join(self.operators_dir, "encodings.pkl")

        # AGREGAR ESTO PARA DEBUG:
        print(f"🔍 Buscando encodings en: {encodings_file}")
        print(f"   - ¿Existe?: {os.path.exists(encodings_file)}")

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
        """Identifica al operador en el frame actual"""
        if not self.known_face_encodings:
            return None
            
        # Detectar condiciones de iluminación para modo día/noche
        self._detect_lighting_conditions(frame)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if not face_locations:
            # Si no hay rostro y había un operador antes, reproducir audio de no registrado
            if self.ultimo_operador_id is not None:
                self.ultimo_operador_id = None
                self.reproducir_audio("assets/audio/no_registrado.mp3")
            return None

        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=self.tolerance
                )

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    operator_id = self.known_face_ids[best_match_index]
                    operator_info = self.operators[operator_id].copy()
                    operator_info['confidence'] = 1 - face_distances[best_match_index]

                    top, right, bottom, left = face_locations[0]
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    operator_info['face_location'] = (top, right, bottom, left)
                    
                    # Obtener landmarks faciales completos para análisis
                    full_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Asegurarse de que estamos buscando landmarks en el área del rostro detectado
                    face_image = full_rgb_frame[top:bottom, left:right]
                    
                    # Solo buscar landmarks si la imagen del rostro es válida
                    if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                        face_landmarks_list = face_recognition.face_landmarks(full_rgb_frame)
                        
                        if face_landmarks_list:
                            operator_info['face_landmarks'] = face_landmarks_list[0]
                            
                            # Analizar expresión facial
                            expresion, expr_confidence = self.analyze_expression(frame, operator_info['face_landmarks'])
                            operator_info['expression'] = expresion
                            operator_info['expression_confidence'] = expr_confidence
                            
                            # Detectar nivel de fatiga
                            fatigue_percentage = self.detect_fatigue(frame, operator_info['face_landmarks'])
                            operator_info['fatigue_percentage'] = fatigue_percentage
                            
                            # Analizar nivel de estrés
                            stress_percentage = self.analyze_stress(frame, operator_info['face_landmarks'])
                            operator_info['stress_percentage'] = stress_percentage

                    # Reproducir audio solo si es un nuevo operador
                    if self.ultimo_operador_id != operator_id:
                        self.reproducir_audio("assets/audio/bienvenido.mp3")
                        self.ultimo_operador_id = operator_id

                    return operator_info

            # Si nadie fue reconocido
            if self.ultimo_operador_id is not None:
                self.ultimo_operador_id = None
                self.reproducir_audio("assets/audio/no_registrado.mp3")

        except Exception as e:
            self.logger.error(f"Error en reconocimiento facial: {str(e)}")

        return None
        
    def _detect_lighting_conditions(self, frame):
        """Detecta las condiciones de iluminación y determina si es modo nocturno"""
        # Convertir a escala de grises si no lo está
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Calcular nivel promedio de iluminación (0-255)
        self.light_level = np.mean(gray)
        
        # Determinar si estamos en modo nocturno
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.night_mode_threshold
        
        # Notificar cambio de modo solo en el log (no en pantalla)
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO" if self.is_night_mode else "DIURNO"
            self.logger.info(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")

    def _draw_text_with_background(self, frame, text, position, font_face, font_scale, color, thickness):
        """Dibuja texto con fondo negro semitransparente"""
        # Obtener tamaño del texto
        (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        
        # Crear una copia del frame para dibujar el fondo semitransparente
        overlay = frame.copy()
        
        # Dibujar rectángulo de fondo
        bg_padding = 5
        bg_start = (position[0] - bg_padding, position[1] - text_height - bg_padding)
        bg_end = (position[0] + text_width + bg_padding, position[1] + bg_padding)
        
        # Dibujar rectángulo negro en la capa de overlay
        cv2.rectangle(overlay, bg_start, bg_end, (0, 0, 0), -1)
        
        # Mezclar el overlay con el frame original (0.5 para semitransparencia)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Dibujar el texto en primer plano
        cv2.putText(frame, text, position, font_face, font_scale, color, thickness)
        
    def draw_operator_info(self, frame, operator_info):
        """Dibuja información del operador en el frame, incluyendo expresiones"""
        # Seleccionar colores según el modo (día/noche)
        colors = self.night_colors if self.is_night_mode else self.day_colors
        
        if operator_info and 'face_location' in operator_info:
            top, right, bottom, left = operator_info['face_location']
            
            # Dibujar puntos faciales (excluyendo ojos)
            if 'face_landmarks' in operator_info:
                landmarks = operator_info['face_landmarks']
                for feature, puntos in landmarks.items():
                    # Saltarse los ojos ("left_eye" y "right_eye")
                    if feature == "left_eye" or feature == "right_eye":
                        continue
                        
                    # Dibujar los demás puntos faciales
                    for (x, y) in puntos:
                        cv2.circle(frame, (x, y), 2, colors['landmarks'], -1)
            else:
                # Si no se encontraron landmarks en identify_operator, intentar obtenerlos aquí
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
                for landmarks in face_landmarks_list:
                    for feature, puntos in landmarks.items():
                        # Saltarse los ojos ("left_eye" y "right_eye")
                        if feature == "left_eye" or feature == "right_eye":
                            continue
                            
                        # Dibujar los demás puntos faciales
                        for (x, y) in puntos:
                            cv2.circle(frame, (x, y), 2, colors['landmarks'], -1)

            # Dibujar nombre al costado del rostro
            name_text = operator_info['name']
            self._draw_text_with_background(frame, name_text, (right + 10, top + 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['name'], 2)
                        
            # Dibujar nivel de confianza (opcional)
            confidence_text = f"Confianza: {operator_info['confidence']:.2f}"
            self._draw_text_with_background(frame, confidence_text, (right + 10, top + 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['name'], 1)
            
            # Dibujar información de expresiones si está disponible
            if 'expression' in operator_info:
                expression_text = f"Expresion: {operator_info['expression']}"
                self._draw_text_with_background(frame, expression_text, (right + 10, top + 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Dibujar información de fatiga si está disponible
            if 'fatigue_percentage' in operator_info:
                fatigue_text = f"Fatiga: {operator_info['fatigue_percentage']}%"
                self._draw_text_with_background(frame, fatigue_text, (right + 10, top + 120),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Dibujar información de estrés si está disponible
            if 'stress_percentage' in operator_info:
                stress_text = f"Estres: {operator_info['stress_percentage']}%"
                self._draw_text_with_background(frame, stress_text, (right + 10, top + 150),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Mensaje si no se reconoce operador
            self._draw_text_with_background(frame, "Operador no reconocido", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    # Métodos reales para análisis de expresiones, fatiga y estrés
    def analyze_expression(self, frame, face_landmarks):
        """Analiza la expresión facial basada en landmarks"""
        if not self.expresion_enabled or not face_landmarks:
            return "neutral", 0.0
        
        try:
            # 1. Calcular características faciales relevantes
            
            # Distancia entre las esquinas de la boca
            mouth_width = self._distance_between_points(
                face_landmarks['top_lip'][0],  # Esquina izquierda
                face_landmarks['top_lip'][6]   # Esquina derecha
            )
            
            # Distancia vertical entre labios
            lip_distance = self._average_distance_between_curves(
                face_landmarks['top_lip'],
                face_landmarks['bottom_lip']
            )
            
            # Altura de la boca
            mouth_height = self._distance_between_points(
                self._midpoint(face_landmarks['top_lip'][3], face_landmarks['top_lip'][4]), 
                self._midpoint(face_landmarks['bottom_lip'][3], face_landmarks['bottom_lip'][4])
            )
            
            # Relación de aspecto de la boca
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Distancia entre cejas y ojos
            eyebrow_eye_dist_left = self._distance_between_points(
                self._midpoint(face_landmarks['left_eyebrow'][2], face_landmarks['left_eyebrow'][3]),
                self._midpoint(face_landmarks['left_eye'][1], face_landmarks['left_eye'][2])
            )
            eyebrow_eye_dist_right = self._distance_between_points(
                self._midpoint(face_landmarks['right_eyebrow'][2], face_landmarks['right_eyebrow'][3]),
                self._midpoint(face_landmarks['right_eye'][1], face_landmarks['right_eye'][2])
            )
            eyebrow_eye_dist = (eyebrow_eye_dist_left + eyebrow_eye_dist_right) / 2
            
            # Ancho de la cara para normalizar
            face_width = self._distance_between_points(
                face_landmarks['chin'][0],  # Lado izquierdo de la barbilla
                face_landmarks['chin'][16]  # Lado derecho de la barbilla
            )
            
            # Normalizar medidas
            norm_mouth_aspect_ratio = mouth_aspect_ratio / face_width * 100
            norm_lip_distance = lip_distance / face_width * 100
            norm_eyebrow_eye_dist = eyebrow_eye_dist / face_width * 100
            
            # 2. Determinar expresión basada en características
            expression = "neutral"
            confidence = 0.6  # Confianza por defecto
            
            # Detectar sonrisa
            if norm_mouth_aspect_ratio < 0.5 and norm_lip_distance < 1.0:
                expression = "sonrisa"
                confidence = 0.8
            
            # Detectar sorpresa
            elif norm_eyebrow_eye_dist > 1.5 and norm_mouth_aspect_ratio > 0.7:
                expression = "sorpresa"
                confidence = 0.75
            
            # Detectar tristeza
            elif norm_eyebrow_eye_dist < 0.8 and norm_mouth_aspect_ratio < 0.4:
                expression = "tristeza"
                confidence = 0.65
            
            # Detectar enojo
            elif norm_eyebrow_eye_dist < 0.7:
                expression = "enojo"
                confidence = 0.7
                
            # 3. Filtrar resultados usando historial para estabilidad
            self.expresion_history.append(expression)
            
            # Determinar expresión estable basada en historial
            self.current_expresion = self._get_stable_expression()
            
            return self.current_expresion, confidence
        except Exception as e:
            self.logger.error(f"Error al analizar expresión: {str(e)}")
            return "neutral", 0.0
    
    def _get_stable_expression(self):
        """Obtiene expresión estable basada en historial"""
        if not self.expresion_history:
            return "neutral"
        
        # Contamos ocurrencias de cada expresión
        counts = {}
        for expr in self.expresion_history:
            counts[expr] = counts.get(expr, 0) + 1
        
        # Encontrar la expresión más común
        most_common = max(counts.items(), key=lambda x: x[1])
        
        # Solo devolver la expresión si aparece en al menos 40% del historial
        if most_common[1] / len(self.expresion_history) >= 0.4:
            return most_common[0]
        else:
            return "neutral"  # Default si no hay consenso
    
    def detect_fatigue(self, frame, face_landmarks):
        """Detecta nivel de fatiga basado en Eye Aspect Ratio (EAR)"""
        if not self.fatiga_enabled or not face_landmarks:
            return 0
        
        try:
            # 1. Calcular EAR (Eye Aspect Ratio)
            left_ear = self._calculate_ear(face_landmarks['left_eye'])
            right_ear = self._calculate_ear(face_landmarks['right_eye'])
            ear = (left_ear + right_ear) / 2.0
            
            # Almacenar en historial
            self.ear_history.append(ear)
            
            # 2. Detectar parpadeos
            current_time = time.time()
            
            # Detectar si estamos en un parpadeo (EAR bajo)
            is_eye_closed = ear < self.fatiga_ear_threshold
            
            # Si los ojos estaban abiertos y ahora están cerrados, registrar como parpadeo
            if is_eye_closed and self.last_blink_time < current_time - 0.2:  # Mínimo 0.2s entre parpadeos
                self.blink_counter += 1
                self.blink_history.append(current_time)
                self.last_blink_time = current_time
            
            # Limpiar historial antiguo (más de time_window segundos)
            self.blink_history = [t for t in self.blink_history if t > current_time - self.fatiga_time_window]
            
            # 3. Calcular métricas de fatiga
            
            # Frecuencia de parpadeo
            blink_rate = len(self.blink_history) / self.fatiga_time_window * 60  # parpadeos por minuto
            
            # EAR promedio (últimos 30 frames)
            avg_ear = sum(self.ear_history) / len(self.ear_history) if self.ear_history else 1.0
            
            # Calcular nivel de fatiga basado en métricas
            # Fatiga aumenta con blink_rate alto y avg_ear bajo
            base_fatigue = 0
            
            # Componente por parpadeos frecuentes
            if blink_rate > 20:  # Más de 20 parpadeos por minuto indica fatiga
                base_fatigue += min(40, (blink_rate - 20) * 2)
            
            # Componente por ojos semicerrados (EAR bajo)
            if avg_ear < 0.3:
                base_fatigue += min(30, (0.3 - avg_ear) * 100)
            
            # Componente por cambios en la variabilidad del EAR
            if len(self.ear_history) > 5:
                ear_std = np.std(list(self.ear_history))
                if ear_std < 0.02:  # Poca variación indica parpadeos lentos o vista fija
                    base_fatigue += 15
            
            # Limitar a 0-100%
            fatigue_level = max(0, min(100, int(base_fatigue)))
            
            # Guardar el valor
            self.fatiga_percentage = fatigue_level
            
            return fatigue_level
        except Exception as e:
            self.logger.error(f"Error al detectar fatiga: {str(e)}")
            return 0
    
    def analyze_stress(self, frame, face_landmarks):
        """Analiza nivel de estrés basado en micro-movimientos y tensión facial"""
        if not self.estres_enabled or not face_landmarks:
            return 0
        
        try:
            # 1. Calcular tensión facial
            
            # Distancia entre cejas (menor distancia indica tensión)
            eyebrow_distance = self._distance_between_points(
                self._midpoint(face_landmarks['left_eyebrow'][0], face_landmarks['left_eyebrow'][1]),
                self._midpoint(face_landmarks['right_eyebrow'][4], face_landmarks['right_eyebrow'][3])
            )
            
            # Ancho de la cara para normalizar
            face_width = self._distance_between_points(
                face_landmarks['chin'][0],  # Lado izquierdo de la barbilla
                face_landmarks['chin'][16]  # Lado derecho de la barbilla
            )
            
            # Tensión normalizada
            normalized_tension = eyebrow_distance / face_width
            
            # 2. Calcular micro-movimientos
            micro_movement = 0
            
            # Si tenemos landmarks previos, comparar con los actuales
            if self.last_landmarks:
                movement_sum = 0
                point_count = 0
                
                # Comparar cada punto facial con su correspondiente en el frame anterior
                for feature in face_landmarks:
                    if feature in self.last_landmarks:
                        for i, point in enumerate(face_landmarks[feature]):
                            if i < len(self.last_landmarks[feature]):
                                movement = self._distance_between_points(
                                    point,
                                    self.last_landmarks[feature][i]
                                )
                                movement_sum += movement
                                point_count += 1
                
                # Calcular movimiento promedio normalizado
                if point_count > 0:
                    micro_movement = movement_sum / point_count / face_width * 100
            
            # Guardar landmarks actuales para el próximo frame
            self.last_landmarks = face_landmarks.copy()
            
            # Añadir al historial
            self.facial_movement_history.append(micro_movement)
            
            # 3. Calcular nivel de estrés
            
            # Componente por tensión facial (cejas juntas)
            tension_component = max(0, (0.5 - normalized_tension) * 100) if normalized_tension < 0.5 else 0
            
            # Componente por micro-movimientos (nerviosismo)
            movement_avg = sum(self.facial_movement_history) / len(self.facial_movement_history) if self.facial_movement_history else 0
            movement_component = min(50, movement_avg * 5)
            
            # Nivel de estrés total
            stress_level = int(tension_component * 0.6 + movement_component * 0.4)
            stress_level = max(0, min(100, stress_level))
            
            # Guardar el valor
            self.estres_percentage = stress_level
            
            return stress_level
        except Exception as e:
            self.logger.error(f"Error al analizar estrés: {str(e)}")
            return 0
    
    # Métodos auxiliares para cálculos
    def _distance_between_points(self, p1, p2):
        """Calcula la distancia euclídea entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _midpoint(self, p1, p2):
        """Calcula el punto medio entre dos puntos"""
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio (EAR) según paper de Soukupová and Čech (2016)"""
        # Calcular distancias verticales
        A = self._distance_between_points(eye_points[1], eye_points[5])
        B = self._distance_between_points(eye_points[2], eye_points[4])
        
        # Calcular distancia horizontal
        C = self._distance_between_points(eye_points[0], eye_points[3])
        
        # Calcular EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 1.0
        return ear
    
    def _average_distance_between_curves(self, curve1, curve2):
        """Calcula la distancia promedio entre dos curvas de puntos"""
        total_dist = 0
        count = 0
        
        # Recorrer los puntos correspondientes
        for i in range(min(len(curve1), len(curve2))):
            dist = self._distance_between_points(curve1[i], curve2[i])
            total_dist += dist
            count += 1
        
        return total_dist / count if count > 0 else 0